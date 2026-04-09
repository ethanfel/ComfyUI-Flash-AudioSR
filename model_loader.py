"""
AudioSR Model Loader node for ComfyUI.

Loads AudioSR (VASR) or FlashSR models with optional auto-download from HuggingFace.
Outputs an AUDIOSR_MODEL dict consumed by AudioSRSampler.

AUDIOSR_MODEL dict schema:
    type:      "vasr" | "flashsr"
    models:    dict of loaded model objects
               vasr:    {"latent_diffusion": <LatentDiffusion>}
               flashsr: {"flashsr_model": <FlashSR>}
    device:    "cuda" | "cpu"
    dtype:     torch.dtype
    cache_key: str  — "{type}:{checkpoint}:{device}:{dtype}"

FlashSR loading (requires sys.path manipulation):
    sys.path.insert(0, flashsr_dir)
    from FlashSR.FlashSR import FlashSR
    model = FlashSR(student_ldm_path, sr_vocoder_path, autoencoder_ckpt_path).to(device)
    model(audio_tensor, num_steps=1, lowpass_input=<bool>)
    # audio_tensor: [channels, 245760] float32 at 48kHz (exactly 5.12s)
"""

import os
import gc
import shutil
import threading
from pathlib import Path

import torch

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

try:
    from comfy import model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

_FLASHSR_HF_REPO = "jakeoneijk/FlashSR_weights"
_FLASHSR_FILES = ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"]

_VASR_HF_REPO = "drbaph/AudioSR"
_VASR_HF_FILES = {
    "basic":  "AudioSR/audiosr_basic_fp32.safetensors",
    "speech": "AudioSR/audiosr_speech_fp32.safetensors",
}
_VASR_LOCAL_FILES = {
    "basic":  "audiosr_basic_fp32.safetensors",
    "speech": "audiosr_speech_fp32.safetensors",
}

_model_cache: dict = {}
_cache_lock = threading.Lock()


def get_model_dir(model_type: str) -> str:
    if HAS_FOLDER_PATHS:
        base = folder_paths.models_dir
    else:
        base = str(Path(__file__).parent.parent / "models")
    if model_type == "flashsr":
        return str(Path(base) / "flashsr")
    return str(Path(base) / "AudioSR")


def build_cache_key(model_type: str, checkpoint: str, device: str, dtype: str) -> str:
    return f"{model_type}:{checkpoint}:{device}:{dtype}"


def _dtype_str_to_torch(dtype: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]


def _download_flashsr_weights(model_dir: str) -> None:
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")
    os.makedirs(model_dir, exist_ok=True)
    for filename in _FLASHSR_FILES:
        dest = Path(model_dir) / filename
        if not dest.exists():
            print(f"[ModelLoader] Downloading {filename} from {_FLASHSR_HF_REPO}...")
            try:
                hf_hub_download(repo_id=_FLASHSR_HF_REPO, filename=filename, local_dir=model_dir)
                print(f"[ModelLoader] Downloaded {filename}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {filename} from {_FLASHSR_HF_REPO}.\n"
                    f"The repo may be private. Set HF_TOKEN env var or place files manually in:\n"
                    f"  {model_dir}\n"
                    f"Required files: {_FLASHSR_FILES}\n"
                    f"Error: {e}"
                )
        else:
            print(f"[ModelLoader] Found {filename}")


def _download_vasr_weights(model_dir: str, variant: str) -> str:
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")
    os.makedirs(model_dir, exist_ok=True)
    hf_filename = _VASR_HF_FILES.get(variant, _VASR_HF_FILES["basic"])
    local_name = _VASR_LOCAL_FILES.get(variant, _VASR_LOCAL_FILES["basic"])
    dest = Path(model_dir) / local_name
    if not dest.exists():
        print(f"[ModelLoader] Downloading {local_name} from {_VASR_HF_REPO}...")
        # hf_filename has subfolder "AudioSR/"; use parent dir so file lands flat in model_dir
        downloaded = hf_hub_download(
            repo_id=_VASR_HF_REPO,
            filename=hf_filename,
            local_dir=str(Path(model_dir).parent),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != dest.resolve():
            shutil.move(str(downloaded_path), str(dest))
        print(f"[ModelLoader] Downloaded {local_name}")
    return str(dest)


def _load_vasr_model(ckpt_path: str, device: str, torch_dtype: torch.dtype) -> dict:
    import sys
    _cur = os.path.dirname(os.path.abspath(__file__))
    if _cur not in sys.path:
        sys.path.insert(0, _cur)

    from versatile_audio_super_resolution.audiosr.utils import default_audioldm_config
    from versatile_audio_super_resolution.audiosr.latent_diffusion.models.ddpm import LatentDiffusion

    config = default_audioldm_config("basic")
    config["model"]["params"]["device"] = device
    model = LatentDiffusion(**config["model"]["params"])

    if ckpt_path.endswith((".safetensors", ".sft")):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path, device="cpu")
        model_dtype = torch.float32
        for v in state_dict.values():
            if v.dtype.is_floating_point:
                model_dtype = v.dtype
                break
        if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}
            model_dtype = torch.float16
        if model_dtype != torch.float32:
            model = model.to(model_dtype)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(torch_dtype).to(device)
    return {"latent_diffusion": model}


def _load_flashsr_model(model_dir: str, device: str, torch_dtype: torch.dtype) -> dict:
    import sys
    flashsr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flashsr")
    if flashsr_dir not in sys.path:
        sys.path.insert(0, flashsr_dir)

    from FlashSR.FlashSR import FlashSR

    ldm_path = str(Path(model_dir) / "student_ldm.pth")
    voc_path = str(Path(model_dir) / "sr_vocoder.pth")
    vae_path = str(Path(model_dir) / "vae.pth")

    for p in [ldm_path, voc_path, vae_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"FlashSR weight not found: {p}\n"
                f"Place student_ldm.pth, sr_vocoder.pth, vae.pth in: {model_dir}"
            )

    print(f"[ModelLoader] Loading FlashSR model components...")
    flashsr_model = FlashSR(ldm_path, voc_path, vae_path)
    flashsr_model.eval()
    flashsr_model = flashsr_model.to(torch_dtype).to(device)
    return {"flashsr_model": flashsr_model}


class AudioSRModelLoader:
    """
    Loads AudioSR or FlashSR model with optional auto-download.
    Output: AUDIOSR_MODEL dict → connect to AudioSRSampler.
    """

    DESCRIPTION = "Load an AudioSR or FlashSR model. Connects to AudioSRSampler."

    @classmethod
    def INPUT_TYPES(cls):
        vasr_dir = get_model_dir("vasr")
        found = []
        if os.path.isdir(vasr_dir):
            found = sorted(
                f for f in os.listdir(vasr_dir)
                if f.endswith((".safetensors", ".pth", ".bin", ".ckpt"))
            )
        checkpoint_options = ["auto-download"] + found if found else ["auto-download", "basic", "speech"]

        return {
            "required": {
                "model_type": (["AudioSR", "FlashSR"], {
                    "default": "AudioSR",
                    "tooltip": "AudioSR: versatile audio super-resolution. FlashSR: fast music-focused upscaler.",
                }),
                "checkpoint": (checkpoint_options, {
                    "default": checkpoint_options[0],
                    "tooltip": "For AudioSR: select checkpoint or auto-download. FlashSR always downloads all 3 weights.",
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device to load model onto.",
                }),
                "dtype": (["fp32", "fp16", "bf16"], {
                    "default": "fp32",
                    "tooltip": "Compute dtype. fp16/bf16 reduce VRAM usage.",
                }),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically download model weights if not present.",
                }),
            }
        }

    RETURN_TYPES = ("AUDIOSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio"

    def load_model(self, model_type: str, checkpoint: str, device: str, dtype: str, auto_download: bool) -> tuple:
        if device == "cuda" and not torch.cuda.is_available():
            print("[ModelLoader] CUDA not available, falling back to CPU")
            device = "cpu"

        torch_dtype = _dtype_str_to_torch(dtype)
        model_dir = get_model_dir(model_type.lower())
        cache_key = build_cache_key(model_type.lower(), checkpoint, device, dtype)

        with _cache_lock:
            if cache_key in _model_cache:
                print(f"[ModelLoader] Using cached model ({cache_key})")
                return (_model_cache[cache_key],)

            print(f"[ModelLoader] Loading {model_type} model ({dtype} on {device})...")

            if model_type == "FlashSR":
                if auto_download:
                    _download_flashsr_weights(model_dir)
                models = _load_flashsr_model(model_dir, device, torch_dtype)
            else:
                if checkpoint == "auto-download":
                    variant = "basic"
                    ckpt_path = str(Path(model_dir) / _VASR_LOCAL_FILES["basic"])
                    if not os.path.exists(ckpt_path):
                        if auto_download:
                            ckpt_path = _download_vasr_weights(model_dir, variant)
                        else:
                            raise FileNotFoundError(
                                f"Model not found: {ckpt_path}\nEnable auto_download or place the file manually."
                            )
                elif checkpoint in _VASR_LOCAL_FILES:
                    variant = checkpoint
                    ckpt_path = str(Path(model_dir) / _VASR_LOCAL_FILES[variant])
                    if not os.path.exists(ckpt_path):
                        if auto_download:
                            ckpt_path = _download_vasr_weights(model_dir, variant)
                        else:
                            raise FileNotFoundError(
                                f"Model not found: {ckpt_path}\nEnable auto_download or place the file manually."
                            )
                else:
                    ckpt_path = str(Path(model_dir) / checkpoint)
                    if not os.path.exists(ckpt_path):
                        raise FileNotFoundError(f"Model not found: {ckpt_path}")
                models = _load_vasr_model(ckpt_path, device, torch_dtype)

            result = {
                "type": model_type.lower(),
                "models": models,
                "device": device,
                "dtype": torch_dtype,
                "cache_key": cache_key,
            }
            _model_cache[cache_key] = result
            print(f"[ModelLoader] Model loaded and cached ({cache_key})")
            return (result,)

    @classmethod
    def IS_CHANGED(cls, model_type, checkpoint, device, dtype, auto_download):
        return build_cache_key(model_type.lower(), checkpoint, device, dtype)
