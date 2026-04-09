"""
Unified AudioSR / FlashSR sampler node for ComfyUI.

Accepts an AUDIOSR_MODEL dict from AudioSRModelLoader and dispatches
to the correct inference backend based on model["type"].
Settings not applicable to the loaded model are silently ignored.
"""

import os
import sys
import gc
import random

import torch
import numpy as np

try:
    from comfy import model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

FLASHSR_CHUNK_SAMPLES = 245760  # 5.12s × 48000 — hard constraint from FlashSR model


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _normalize_audio_input(audio) -> tuple:
    """Accept ComfyUI AUDIO (dict or legacy tuple). Return (waveform_np [C, S], sr)."""
    if isinstance(audio, dict):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
    else:
        waveform, sr = audio
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    return waveform.astype(np.float32), int(sr)


def _make_audio_output(waveform_np: np.ndarray, sr: int) -> dict:
    """Convert [C, S] numpy array to ComfyUI AUDIO dict with shape [1, C, S]."""
    t = torch.from_numpy(waveform_np)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    t = t.unsqueeze(0)
    return {"waveform": t, "sample_rate": sr}


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    import librosa
    return np.stack(
        [librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr) for ch in waveform],
        axis=0,
    )


def _check_interrupted():
    if HAS_COMFY:
        model_management.throw_exception_if_processing_interrupted()


# ---------------------------------------------------------------------------
# AudioSR (VASR) inference
# ---------------------------------------------------------------------------

def _run_vasr(waveform, sr, model_obj, ddim_steps, guidance_scale, seed, chunk_size, overlap, attention_backend, dtype):
    """Run VASR inference. Returns [C, S] numpy at 48kHz."""
    from versatile_audio_super_resolution.audiosr.pipeline import make_batch_for_super_resolution, seed_everything
    from versatile_audio_super_resolution.audiosr.latent_diffusion.modules.attention import set_attention_backend, set_attention_dtype

    attention_dtype = None if dtype == "fp32" else dtype
    if attention_backend == "sageattn" and dtype == "fp32":
        print("[Sampler] SageAttention requires fp16/bf16 — falling back to sdpa")
        attention_backend = "sdpa"
    set_attention_backend(attention_backend)
    set_attention_dtype(attention_dtype)

    if seed == 0:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    waveform = _resample(waveform, sr, 48000)
    sr = 48000
    latent_diffusion = model_obj["latent_diffusion"]
    num_samples = waveform.shape[1]
    duration_sec = num_samples / sr

    processed = []
    for channel in [waveform[i] for i in range(waveform.shape[0])]:
        _check_interrupted()
        if duration_sec <= 10.24:
            batch, _ = make_batch_for_super_resolution(None, waveform=np.expand_dims(channel, 0))
            out = latent_diffusion.generate_batch(batch, unconditional_guidance_scale=guidance_scale, ddim_steps=ddim_steps, duration=duration_sec)
            if isinstance(out, np.ndarray):
                out = torch.from_numpy(out)
            if out.ndim == 1:
                out = out.unsqueeze(0)
            target = int(duration_sec * 48000)
            if out.shape[-1] > target:
                out = out[..., :target]
            processed.append(out.squeeze(0).cpu().numpy())
        else:
            chunk_s = int(chunk_size * sr)
            overlap_s = int(overlap * sr)
            out_overlap_s = int(overlap * 48000)
            MIN_CHUNK = int(5.12 * sr)
            chunks = []
            start = 0
            while start < num_samples:
                end = min(start + chunk_s, num_samples)
                chunk = channel[start:end]
                is_padded = len(chunk) < MIN_CHUNK
                if is_padded:
                    chunk = np.pad(chunk, (0, MIN_CHUNK - len(chunk)), mode="constant")
                chunks.append((chunk, start, end, is_padded))
                start += chunk_s - overlap_s

            reconstructed = np.zeros(int(duration_sec * 48000))
            weight_sum = np.zeros(int(duration_sec * 48000))

            for i, (chunk, orig_start, orig_end, is_padded) in enumerate(chunks):
                _check_interrupted()
                batch, _ = make_batch_for_super_resolution(None, waveform=np.expand_dims(chunk, 0))
                out = latent_diffusion.generate_batch(batch, unconditional_guidance_scale=guidance_scale, ddim_steps=ddim_steps, duration=len(chunk) / sr)
                if isinstance(out, np.ndarray):
                    out = torch.from_numpy(out)
                if out.ndim == 1:
                    out = out.unsqueeze(0)
                elif out.ndim > 2:
                    out = out.squeeze()
                out_np = out.squeeze(0).cpu().numpy()
                if out_np.ndim != 1:
                    out_np = out_np.flatten()

                out_start = int(orig_start / sr * 48000)
                expected_len = int((orig_end - orig_start) / sr * 48000)
                out_end = min(out_start + expected_len, reconstructed.shape[0])
                slice_len = out_end - out_start

                if is_padded:
                    out_np = out_np[:expected_len]
                if out_np.shape[0] > slice_len:
                    out_np = out_np[:slice_len]
                elif out_np.shape[0] < slice_len:
                    padded = np.zeros(slice_len)
                    padded[:out_np.shape[0]] = out_np
                    out_np = padded

                weights = np.ones(slice_len)
                if overlap > 0 and i < len(chunks) - 1:
                    fl = min(out_overlap_s, slice_len)
                    out_np[-fl:] *= np.linspace(1.0, 0.0, fl)
                    weights[-fl:] *= np.linspace(1.0, 0.0, fl)
                if overlap > 0 and i > 0:
                    fl = min(out_overlap_s, slice_len)
                    out_np[:fl] *= np.linspace(0.0, 1.0, fl)
                    weights[:fl] *= np.linspace(0.0, 1.0, fl)

                reconstructed[out_start:out_end] += out_np
                weight_sum[out_start:out_end] += weights

            nz = weight_sum > 0
            if np.any(nz):
                reconstructed[nz] /= weight_sum[nz]
            processed.append(reconstructed)

    return np.stack(processed, axis=0)


# ---------------------------------------------------------------------------
# FlashSR inference
# ---------------------------------------------------------------------------

def _run_flashsr(waveform, sr, model_obj, output_sr, lowpass_input, chunk_size, overlap):
    """
    Run FlashSR inference. Returns [C, S] numpy at output_sr.
    FlashSR processes audio in fixed 5.12s (245760 sample) chunks at 48kHz.
    """
    flashsr_dir = os.path.join(_current_dir, "flashsr")
    if flashsr_dir not in sys.path:
        sys.path.insert(0, flashsr_dir)

    flashsr_model = model_obj["flashsr_model"]

    # Resample to 48kHz
    waveform_48k = _resample(waveform, sr, 48000)

    # Optional lowpass before inference (lowpass_input default in FlashSR is True)
    if lowpass_input:
        try:
            from versatile_audio_super_resolution.audiosr.lowpass import lowpass_filter
            waveform_48k = np.stack(
                [lowpass_filter(ch, 48000) for ch in waveform_48k], axis=0
            )
        except Exception as e:
            print(f"[Sampler] lowpass_filter failed ({e}) — skipping")

    # Use FlashSR native chunk size (5.12s = 245760 samples at 48kHz)
    # User chunk_size/overlap are ignored for FlashSR — chunk size is fixed by the model
    num_samples = waveform_48k.shape[1]
    processed = []

    for channel in [waveform_48k[i] for i in range(waveform_48k.shape[0])]:
        _check_interrupted()
        # Pad to multiple of FLASHSR_CHUNK_SAMPLES
        remainder = num_samples % FLASHSR_CHUNK_SAMPLES
        if remainder != 0:
            pad_len = FLASHSR_CHUNK_SAMPLES - remainder
            channel_padded = np.pad(channel, (0, pad_len), mode="constant")
        else:
            channel_padded = channel
            pad_len = 0

        # Split into chunks and run inference
        n_chunks = len(channel_padded) // FLASHSR_CHUNK_SAMPLES
        out_chunks = []
        for i in range(n_chunks):
            chunk = channel_padded[i * FLASHSR_CHUNK_SAMPLES:(i + 1) * FLASHSR_CHUNK_SAMPLES]
            model_param = next(flashsr_model.parameters())
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(dtype=model_param.dtype, device=model_param.device)
            # lowpass_input=False: we handle lowpass ourselves above; don't apply again inside model
            with torch.no_grad():
                out = flashsr_model(chunk_tensor, num_steps=1, lowpass_input=False)
            out_chunks.append(out.squeeze(0).cpu().numpy())

        channel_out = np.concatenate(out_chunks, axis=0)
        # Remove padding
        if pad_len > 0:
            channel_out = channel_out[:num_samples]
        processed.append(channel_out)

    result = np.stack(processed, axis=0)

    # Resample to requested output SR
    if output_sr != 48000:
        result = _resample(result, 48000, output_sr)

    return result


# ---------------------------------------------------------------------------
# Cache eviction (called when unload_model=True)
# ---------------------------------------------------------------------------

def _evict_from_cache(cache_key: str):
    try:
        from .model_loader import _model_cache, _cache_lock
    except ImportError:
        from model_loader import _model_cache, _cache_lock
    with _cache_lock:
        if cache_key in _model_cache:
            entry = _model_cache.pop(cache_key)
            del entry
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Sampler] Model unloaded ({cache_key})")


# ---------------------------------------------------------------------------
# AudioSRSampler node
# ---------------------------------------------------------------------------

class AudioSRSampler:
    """
    Unified audio super-resolution sampler.
    Dispatches to AudioSR or FlashSR based on the connected model.
    Settings not used by the loaded model are silently ignored.
    """

    DESCRIPTION = "Run audio super-resolution using an AudioSR or FlashSR model."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "model": ("AUDIOSR_MODEL", {}),
            },
            "optional": {
                "ddim_steps": ("INT", {"default": 50, "min": 10, "max": 500, "step": 1,
                    "tooltip": "[AudioSR only] Denoising steps"}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "[AudioSR only] Classifier-free guidance scale"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "[AudioSR only] Random seed (0 = random)"}),
                "output_sr": (["48000", "44100", "96000"], {"default": "48000",
                    "tooltip": "[FlashSR only] Target output sample rate"}),
                "lowpass_input": ("BOOLEAN", {"default": False,
                    "tooltip": "[FlashSR only] Apply lowpass filter before inference"}),
                "chunk_size": ("FLOAT", {"default": 10.24, "min": 2.56, "max": 30.0, "step": 0.01,
                    "tooltip": "[AudioSR only] Chunk duration in seconds"}),
                "overlap": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "[AudioSR only] Overlap between chunks in seconds"}),
                "attention_backend": (["sdpa", "sageattn", "eager"], {"default": "sdpa",
                    "tooltip": "[AudioSR only] Attention backend"}),
                "unload_model": ("BOOLEAN", {"default": False,
                    "tooltip": "Unload model from VRAM after generation"}),
                "show_spectrogram": ("BOOLEAN", {"default": True,
                    "tooltip": "Output before/after spectrogram comparison image"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "IMAGE")
    RETURN_NAMES = ("audio", "spectrogram")
    FUNCTION = "run"
    CATEGORY = "audio"

    def run(self, audio, model, ddim_steps=50, guidance_scale=3.5, seed=0,
            output_sr="48000", lowpass_input=False, chunk_size=10.24, overlap=0.5,
            attention_backend="sdpa", unload_model=False, show_spectrogram=True):

        try:
            from .vasr_node import generate_spectrogram_comparison
        except ImportError:
            from vasr_node import generate_spectrogram_comparison

        waveform, sr = _normalize_audio_input(audio)
        original_waveform = waveform.copy()
        original_sr = sr

        model_type = model["type"]
        models = model["models"]
        dtype_str = {
            torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16",
        }.get(model["dtype"], "fp32")

        print(f"[Sampler] Running {model_type} on {waveform.shape[1] / sr:.2f}s audio ({sr}Hz)")

        with torch.no_grad():
            if model_type == "flashsr":
                result = _run_flashsr(waveform, sr, models,
                    output_sr=int(output_sr), lowpass_input=lowpass_input,
                    chunk_size=chunk_size, overlap=overlap)
                out_sr = int(output_sr)
            else:
                result = _run_vasr(waveform, sr, models,
                    ddim_steps=ddim_steps, guidance_scale=guidance_scale, seed=seed,
                    chunk_size=chunk_size, overlap=overlap,
                    attention_backend=attention_backend, dtype=dtype_str)
                out_sr = 48000

        print(f"[Sampler] Done. Output: {result.shape[1] / out_sr:.2f}s at {out_sr}Hz")

        if unload_model:
            _evict_from_cache(model["cache_key"])

        spec_tensor = torch.zeros((1, 256, 256, 3))
        if show_spectrogram:
            img = generate_spectrogram_comparison(original_waveform, original_sr, result, out_sr)
            if img is not None:
                img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                spec_tensor = torch.clamp(torch.from_numpy(arr).unsqueeze(0), 0.0, 1.0)

        return (_make_audio_output(result, out_sr), spec_tensor)
