# FlashSR Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `AudioSRModelLoader` node and a unified `AudioSRSampler` node that supports both AudioSR (VASR) and FlashSR models with auto-download, while keeping the existing `VASRNode` intact for backward compatibility.

**Architecture:** Two new files — `model_loader.py` (handles model selection, auto-download, outputs `AUDIOSR_MODEL` dict) and `sampler_node.py` (unified inference node that dispatches to AudioSR or FlashSR based on loaded model type). FlashSR inference code is bundled as a minimal copy in `flashsr/`, mirroring how `versatile_audio_super_resolution/` is bundled. `__init__.py` registers all nodes.

**Tech Stack:** Python, PyTorch, `huggingface_hub`, `torchaudio`/`librosa` for resampling, ComfyUI node API, existing `versatile_audio_super_resolution` local module.

---

## Key Reference Files

- `vasr_node.py` — existing AudioSR node; do NOT break it; reuse `generate_spectrogram_comparison()` by importing from it
- `versatile_audio_super_resolution/audiosr/pipeline.py` — `make_batch_for_super_resolution`, `seed_everything`
- `versatile_audio_super_resolution/audiosr/latent_diffusion/models/ddpm.py` — `LatentDiffusion`
- `versatile_audio_super_resolution/audiosr/latent_diffusion/modules/attention.py` — `set_attention_backend`, `set_attention_dtype`
- `__init__.py` — node registration; update at the end

## AUDIOSR_MODEL Type

All tasks use this dict structure (never a class instance — keep it a plain dict):

```python
{
    "type": "vasr" | "flashsr",
    "models": {
        # vasr:    {"latent_diffusion": <LatentDiffusion>}
        # flashsr: {"ldm": <model>, "vocoder": <model>, "vae": <model>}
    },
    "device": "cuda" | "cpu",
    "dtype": torch.float32 | torch.float16 | torch.bfloat16,
    "cache_key": str,   # "{type}:{checkpoint}:{device}:{dtype}"
}
```

---

## Task 1: Research and Bundle FlashSR Inference Code

**Files:**
- Create: `flashsr/__init__.py`
- Create: `flashsr/<files from FlashSR_Inference repo>`

**Step 1: Fetch the FlashSR_Inference repo structure**

Run:
```bash
pip show huggingface_hub  # confirm available
python -c "import requests; r = requests.get('https://api.github.com/repos/jakeoneijk/FlashSR_Inference/git/trees/main?recursive=1'); import json; [print(f['path']) for f in json.loads(r.text).get('tree', []) if f['type'] == 'blob' and f['path'].endswith('.py')]"
```

This lists all Python files in the repo. Identify the minimal set needed:
- Model architecture files (e.g. `model.py`, `ldm.py`, `vocoder.py`, `vae.py`)
- Inference entry point (e.g. `inference.py`, `pipeline.py`)
- Config files (`.yaml` or `.json`)

**Step 2: Download the minimal files**

For each file identified as needed, download it:
```bash
curl -L "https://raw.githubusercontent.com/jakeoneijk/FlashSR_Inference/main/<path>" -o "flashsr/<path>"
```

Copy only files directly required for inference — no training code, no demo scripts, no notebooks. If there's a `requirements.txt` in the repo, read it and check if any new deps are needed (add to our `requirements.txt` if so).

**Step 3: Create `flashsr/__init__.py`**

```python
"""
FlashSR inference code — bundled from https://github.com/jakeoneijk/FlashSR_Inference
Minimal copy; only inference-relevant files included.
"""
```

**Step 4: Verify the import works**

```bash
cd /media/p5/ComfyUI-Flash-AudioSR
python -c "import sys; sys.path.insert(0, '.'); from flashsr import <main_inference_class>"
```

Expected: no ImportError. Fix any missing imports by adjusting relative imports or adding missing files.

**Step 5: Identify the inference API**

Read the main inference file and note the exact API. The Egregora reference shows:
- Input: audio as `np.ndarray` at 48 kHz, `[samples]`
- The model processes it in 5.12s chunks (245760 samples at 48kHz) with 0.5s overlap using Hann WOLA
- Three model components loaded from `.pth` files: `student_ldm.pth`, `sr_vocoder.pth`, `vae.pth`

Document the actual API in a comment at the top of `model_loader.py` (Task 2).

**Step 6: Commit**

```bash
git add flashsr/
git commit -m "feat: bundle FlashSR inference code"
```

---

## Task 2: Create `model_loader.py`

**Files:**
- Create: `model_loader.py`
- Create: `tests/test_model_loader.py`

### Model directory layout

```
ComfyUI/models/AudioSR/       ← VASR checkpoints (.safetensors, .pth, .bin)
ComfyUI/models/flashsr/       ← FlashSR weights (student_ldm.pth, sr_vocoder.pth, vae.pth)
```

### Step 0: Create tests directory

```bash
mkdir -p tests && touch tests/__init__.py
```

### Step 1: Write the failing tests

Create `tests/test_model_loader.py`:

```python
"""Tests for AudioSRModelLoader node."""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import get_model_dir, build_cache_key, AudioSRModelLoader


def test_get_model_dir_vasr(tmp_path, monkeypatch):
    """get_model_dir returns a path ending in AudioSR for vasr."""
    result = get_model_dir("vasr")
    assert result.endswith("AudioSR") or "AudioSR" in result


def test_get_model_dir_flashsr(tmp_path, monkeypatch):
    """get_model_dir returns a path ending in flashsr for flashsr."""
    result = get_model_dir("flashsr")
    assert result.endswith("flashsr") or "flashsr" in result


def test_build_cache_key():
    cache_key = build_cache_key("vasr", "model.safetensors", "cuda", "fp32")
    assert cache_key == "vasr:model.safetensors:cuda:fp32"


def test_build_cache_key_flashsr():
    cache_key = build_cache_key("flashsr", "auto", "cpu", "fp16")
    assert cache_key == "flashsr:auto:cpu:fp16"


def test_input_types_structure():
    """INPUT_TYPES returns required keys."""
    inputs = AudioSRModelLoader.INPUT_TYPES()
    assert "required" in inputs
    required = inputs["required"]
    assert "model_type" in required
    assert "device" in required
    assert "dtype" in required
    assert "auto_download" in required


def test_return_types():
    assert AudioSRModelLoader.RETURN_TYPES == ("AUDIOSR_MODEL",)
```

**Step 2: Run tests to verify they fail**

```bash
cd /media/p5/ComfyUI-Flash-AudioSR
python -m pytest tests/test_model_loader.py -v 2>&1 | head -40
```

Expected: `ModuleNotFoundError: No module named 'model_loader'`

**Step 3: Create `model_loader.py`**

```python
"""
AudioSR Model Loader node for ComfyUI.

Loads AudioSR (VASR) or FlashSR models with optional auto-download from HuggingFace.
Outputs an AUDIOSR_MODEL dict consumed by AudioSRSampler.

AUDIOSR_MODEL dict schema:
    type:      "vasr" | "flashsr"
    models:    dict of loaded model objects
    device:    "cuda" | "cpu"
    dtype:     torch.dtype
    cache_key: str  — "{type}:{checkpoint}:{device}:{dtype}"
"""

import os
import gc
import threading
from pathlib import Path

import torch

# ComfyUI integration
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

# HuggingFace download
try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

_FLASHSR_HF_REPO = "jakeoneijk/FlashSR_weights"  # may be private — see NOTE below
_FLASHSR_FILES = ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"]

# NOTE: jakeoneijk/FlashSR_weights returned 404 during planning. If it's private/gated,
# auto-download will fail with a clear error. Users must either provide an HF token
# (set HF_TOKEN env var) or place the files manually in ComfyUI/models/flashsr/.
# Verify the correct repo name before implementing by checking the Egregora plugin source.

_VASR_HF_REPO = "drbaph/AudioSR"
# Files live in an AudioSR/ subfolder inside the repo:
_VASR_HF_FILES = {
    "basic":  "AudioSR/audiosr_basic_fp32.safetensors",
    "speech": "AudioSR/audiosr_speech_fp32.safetensors",
}
# Local names after download (flat — no subfolder in our models dir):
_VASR_LOCAL_FILES = {
    "basic":  "audiosr_basic_fp32.safetensors",
    "speech": "audiosr_speech_fp32.safetensors",
}

# Global model cache — keyed by cache_key
_model_cache: dict = {}
_cache_lock = threading.Lock()


def get_model_dir(model_type: str) -> str:
    """Return the model directory path for the given model type."""
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
    """Download all three FlashSR weight files from HuggingFace if missing."""
    if not HAS_HF_HUB:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )
    os.makedirs(model_dir, exist_ok=True)
    for filename in _FLASHSR_FILES:
        dest = Path(model_dir) / filename
        if not dest.exists():
            print(f"[ModelLoader] Downloading {filename} from {_FLASHSR_HF_REPO}...")
            hf_hub_download(
                repo_id=_FLASHSR_HF_REPO,
                filename=filename,
                local_dir=model_dir,
            )
            print(f"[ModelLoader] Downloaded {filename}")
        else:
            print(f"[ModelLoader] Found {filename}")


def _download_vasr_weights(model_dir: str, variant: str) -> str:
    """Download the specified VASR checkpoint and return its local path."""
    if not HAS_HF_HUB:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )
    os.makedirs(model_dir, exist_ok=True)
    hf_filename = _VASR_HF_FILES.get(variant, _VASR_HF_FILES["basic"])
    local_name = _VASR_LOCAL_FILES.get(variant, _VASR_LOCAL_FILES["basic"])
    dest = Path(model_dir) / local_name
    if not dest.exists():
        print(f"[ModelLoader] Downloading {local_name} from {_VASR_HF_REPO}...")
        # hf_filename is "AudioSR/foo.safetensors"; use parent of model_dir as local_dir
        # so the file lands at model_dir/foo.safetensors (not model_dir/AudioSR/foo.safetensors)
        downloaded = hf_hub_download(
            repo_id=_VASR_HF_REPO,
            filename=hf_filename,
            local_dir=str(Path(model_dir).parent),
        )
        # hf_hub_download returns the actual path; move to flat location if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != dest:
            import shutil
            shutil.move(str(downloaded_path), str(dest))
        print(f"[ModelLoader] Downloaded {local_name}")
    return str(dest)


def _load_vasr_model(ckpt_path: str, device: str, torch_dtype: torch.dtype) -> dict:
    """Load VASR LatentDiffusion model. Returns models dict."""
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
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(torch_dtype).to(device)
    return {"latent_diffusion": model}


def _load_flashsr_model(model_dir: str, device: str, torch_dtype: torch.dtype) -> dict:
    """Load the three FlashSR model components. Returns models dict."""
    import sys
    _cur = os.path.dirname(os.path.abspath(__file__))
    if _cur not in sys.path:
        sys.path.insert(0, _cur)

    # Import from bundled flashsr module — adjust class names after Task 1
    from flashsr import load_student_ldm, load_vocoder, load_vae  # adjust to actual API

    ldm_path = str(Path(model_dir) / "student_ldm.pth")
    voc_path = str(Path(model_dir) / "sr_vocoder.pth")
    vae_path = str(Path(model_dir) / "vae.pth")

    for p in [ldm_path, voc_path, vae_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"FlashSR weight not found: {p}")

    ldm = load_student_ldm(ldm_path, device=device, dtype=torch_dtype)
    vocoder = load_vocoder(voc_path, device=device, dtype=torch_dtype)
    vae = load_vae(vae_path, device=device, dtype=torch_dtype)

    return {"ldm": ldm, "vocoder": vocoder, "vae": vae}


class AudioSRModelLoader:
    """
    Loads AudioSR or FlashSR model with optional auto-download.
    Output is an AUDIOSR_MODEL dict for use with AudioSRSampler.
    """

    DESCRIPTION = "Load an AudioSR or FlashSR model. Connects to AudioSRSampler."

    @classmethod
    def INPUT_TYPES(cls):
        # Scan AudioSR model dir for existing checkpoints; prepend auto-download
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
                    "tooltip": "For AudioSR: select a checkpoint or use auto-download. FlashSR always downloads all 3 weights.",
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

    def load_model(
        self,
        model_type: str,
        checkpoint: str,
        device: str,
        dtype: str,
        auto_download: bool,
    ) -> tuple:
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
                # AudioSR / VASR
                if checkpoint == "auto-download":
                    # Default to basic; download if needed
                    variant = "basic"
                    ckpt_path = str(Path(model_dir) / _VASR_LOCAL_FILES["basic"])
                    if not os.path.exists(ckpt_path):
                        if auto_download:
                            ckpt_path = _download_vasr_weights(model_dir, variant)
                        else:
                            raise FileNotFoundError(
                                f"Model not found: {ckpt_path}\n"
                                f"Enable auto_download or place the file manually."
                            )
                elif checkpoint in _VASR_LOCAL_FILES:
                    # Named variant (basic / speech) — may need download
                    variant = checkpoint
                    ckpt_path = str(Path(model_dir) / _VASR_LOCAL_FILES[variant])
                    if not os.path.exists(ckpt_path):
                        if auto_download:
                            ckpt_path = _download_vasr_weights(model_dir, variant)
                        else:
                            raise FileNotFoundError(
                                f"Model not found: {ckpt_path}\n"
                                f"Enable auto_download or place the file manually."
                            )
                else:
                    # User selected a scanned file from the model dir
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
```

**Step 4: Run tests**

```bash
cd /media/p5/ComfyUI-Flash-AudioSR
python -m pytest tests/test_model_loader.py -v
```

Expected: all 6 tests pass. Fix any failures.

> NOTE: `_load_flashsr_model` imports from `flashsr` — tests don't call it directly, so the import isn't exercised. The actual import is verified in Task 1 Step 4.

**Step 5: Commit**

```bash
git add model_loader.py tests/test_model_loader.py
git commit -m "feat: add AudioSRModelLoader node with auto-download"
```

---

## Task 3: Create `sampler_node.py`

**Files:**
- Create: `sampler_node.py`
- Create: `tests/test_sampler_node.py`

### Step 1: Write the failing tests

Create `tests/test_sampler_node.py`:

```python
"""Tests for AudioSRSampler node."""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampler_node import AudioSRSampler, _normalize_audio_input, _make_audio_output


def make_audio(sr=22050, duration=1.0, channels=1):
    """Helper: create a ComfyUI AUDIO dict."""
    samples = int(sr * duration)
    waveform = torch.randn(1, channels, samples)
    return {"waveform": waveform, "sample_rate": sr}


def test_input_types_structure():
    inputs = AudioSRSampler.INPUT_TYPES()
    assert "required" in inputs
    required = inputs["required"]
    assert "audio" in required
    assert "model" in required
    optional = inputs.get("optional", {})
    assert "ddim_steps" in optional
    assert "output_sr" in optional
    assert "chunk_size" in optional


def test_return_types():
    assert "AUDIO" in AudioSRSampler.RETURN_TYPES
    assert "IMAGE" in AudioSRSampler.RETURN_TYPES


def test_normalize_audio_dict():
    """_normalize_audio_input handles dict format."""
    audio = make_audio(sr=44100, duration=0.5, channels=1)
    waveform, sr = _normalize_audio_input(audio)
    assert sr == 44100
    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 2  # [channels, samples]


def test_normalize_audio_stereo():
    """_normalize_audio_input handles stereo."""
    audio = make_audio(sr=48000, duration=0.5, channels=2)
    waveform, sr = _normalize_audio_input(audio)
    assert waveform.shape[0] == 2


def test_normalize_audio_tuple():
    """_normalize_audio_input handles legacy tuple format."""
    samples = int(22050 * 0.5)
    waveform_t = torch.randn(1, 1, samples)
    audio = (waveform_t, 22050)
    waveform, sr = _normalize_audio_input(audio)
    assert sr == 22050
    assert isinstance(waveform, np.ndarray)


def test_make_audio_output():
    """_make_audio_output produces ComfyUI AUDIO dict."""
    waveform = np.random.randn(2, 48000).astype(np.float32)
    result = _make_audio_output(waveform, 48000)
    assert "waveform" in result
    assert "sample_rate" in result
    assert result["sample_rate"] == 48000
    assert result["waveform"].shape == (1, 2, 48000)  # [B, C, S]
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sampler_node.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'sampler_node'`

**Step 3: Create `sampler_node.py`**

```python
"""
Unified AudioSR / FlashSR sampler node for ComfyUI.

Accepts an AUDIOSR_MODEL dict from AudioSRModelLoader and dispatches
to the correct inference backend based on model["type"].

Settings not applicable to the loaded model are silently ignored.
"""

import os
import sys
import gc
import threading
import random

import torch
import numpy as np

# ComfyUI integration
try:
    from comfy import model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


# ---------------------------------------------------------------------------
# Shared audio helpers
# ---------------------------------------------------------------------------

def _normalize_audio_input(audio) -> tuple[np.ndarray, int]:
    """
    Accept ComfyUI AUDIO (dict or legacy tuple). Return (waveform_np, sr).
    waveform_np shape: [channels, samples], float32.
    """
    if isinstance(audio, dict):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
    else:
        waveform, sr = audio

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()

    # Remove batch dim: [B, C, S] → [C, S]
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
    # Add channel dim if missing: [S] → [1, S]
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]

    return waveform.astype(np.float32), int(sr)


def _make_audio_output(waveform_np: np.ndarray, sr: int) -> dict:
    """
    Convert [channels, samples] numpy array to ComfyUI AUDIO dict.
    Output waveform shape: [1, channels, samples].
    """
    t = torch.from_numpy(waveform_np)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    t = t.unsqueeze(0)  # add batch dim
    return {"waveform": t, "sample_rate": sr}


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample each channel from orig_sr to target_sr."""
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
# AudioSR (VASR) inference path
# ---------------------------------------------------------------------------

def _run_vasr(
    waveform: np.ndarray,
    sr: int,
    model_obj,
    ddim_steps: int,
    guidance_scale: float,
    seed: int,
    chunk_size: float,
    overlap: float,
    attention_backend: str,
    dtype: str,
) -> np.ndarray:
    """Run VASR inference. Returns [channels, samples] at 48kHz."""
    from versatile_audio_super_resolution.audiosr.pipeline import (
        make_batch_for_super_resolution,
        seed_everything,
    )
    from versatile_audio_super_resolution.audiosr.latent_diffusion.modules.attention import (
        set_attention_backend,
        set_attention_dtype,
    )

    attention_dtype = None if dtype == "fp32" else dtype
    if attention_backend == "sageattn" and dtype == "fp32":
        print("[Sampler] SageAttention requires fp16/bf16 — falling back to sdpa")
        attention_backend = "sdpa"

    set_attention_backend(attention_backend)
    set_attention_dtype(attention_dtype)

    if seed == 0:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    # Resample to 48kHz
    waveform = _resample(waveform, sr, 48000)
    sr = 48000

    latent_diffusion = model_obj["latent_diffusion"]
    num_samples = waveform.shape[1]
    duration_sec = num_samples / sr
    channels = [waveform[i] for i in range(waveform.shape[0])]

    processed = []
    for ch_idx, channel in enumerate(channels):
        _check_interrupted()

        if duration_sec <= 10.24:
            batch, _ = make_batch_for_super_resolution(
                None, waveform=np.expand_dims(channel, 0)
            )
            out = latent_diffusion.generate_batch(
                batch,
                unconditional_guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                duration=duration_sec,
            )
            if isinstance(out, np.ndarray):
                out = torch.from_numpy(out)
            if out.ndim == 1:
                out = out.unsqueeze(0)
            target = int(duration_sec * 48000)
            if out.shape[-1] > target:
                out = out[..., :target]
            processed.append(out.squeeze(0).cpu().numpy())
        else:
            chunk_samples = int(chunk_size * sr)
            overlap_samples = int(overlap * sr)
            output_overlap_samples = int(overlap * 48000)
            MIN_CHUNK = int(5.12 * sr)

            chunks = []
            start = 0
            while start < num_samples:
                end = min(start + chunk_samples, num_samples)
                chunk = channel[start:end]
                is_padded = len(chunk) < MIN_CHUNK
                if is_padded:
                    chunk = np.pad(chunk, (0, MIN_CHUNK - len(chunk)), mode="constant")
                chunks.append((chunk, start, end, is_padded))
                start += chunk_samples - overlap_samples

            reconstructed = np.zeros(int(duration_sec * 48000))
            weight_sum = np.zeros(int(duration_sec * 48000))

            for i, (chunk, orig_start, orig_end, is_padded) in enumerate(chunks):
                _check_interrupted()
                chunk_dur = len(chunk) / sr
                batch, _ = make_batch_for_super_resolution(
                    None, waveform=np.expand_dims(chunk, 0)
                )
                out = latent_diffusion.generate_batch(
                    batch,
                    unconditional_guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    duration=chunk_dur,
                )
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
                    padded[: out_np.shape[0]] = out_np
                    out_np = padded

                weights = np.ones(slice_len)
                if overlap > 0 and i < len(chunks) - 1:
                    fl = min(output_overlap_samples, slice_len)
                    fade = np.linspace(1.0, 0.0, fl)
                    out_np[-fl:] *= fade
                    weights[-fl:] *= fade
                if overlap > 0 and i > 0:
                    fl = min(output_overlap_samples, slice_len)
                    fade = np.linspace(0.0, 1.0, fl)
                    out_np[:fl] *= fade
                    weights[:fl] *= fade

                reconstructed[out_start:out_end] += out_np
                weight_sum[out_start:out_end] += weights

            nz = weight_sum > 0
            if np.any(nz):
                reconstructed[nz] /= weight_sum[nz]
            processed.append(reconstructed)

    return np.stack(processed, axis=0)


# ---------------------------------------------------------------------------
# FlashSR inference path
# ---------------------------------------------------------------------------

FLASHSR_CHUNK_SAMPLES = 245760   # 5.12s × 48000
FLASHSR_OVERLAP_SAMPLES = 24000  # 0.5s × 48000


def _run_flashsr(
    waveform: np.ndarray,
    sr: int,
    model_obj: dict,
    output_sr: int,
    lowpass_input: bool,
    chunk_size: float,
    overlap: float,
) -> np.ndarray:
    """
    Run FlashSR inference.
    Returns [channels, samples] at output_sr.

    NOTE: Update the import and call signature below to match the actual
    FlashSR inference API discovered in Task 1 Step 5.
    """
    from flashsr import run_inference  # adjust to actual API

    # Resample to 48kHz (FlashSR internal requirement)
    waveform_48k = _resample(waveform, sr, 48000)

    # Optional lowpass before inference
    # lowpass.py exports: lowpass_filter(audio, cutoff_freq, sample_rate) — verify exact
    # signature from versatile_audio_super_resolution/audiosr/lowpass.py before using.
    if lowpass_input:
        try:
            from versatile_audio_super_resolution.audiosr.lowpass import lowpass_filter
            import inspect
            sig = inspect.signature(lowpass_filter)
            params = list(sig.parameters.keys())
            # Call with positional args to avoid kwarg name assumptions
            waveform_48k = np.stack(
                [lowpass_filter(ch, 48000) for ch in waveform_48k], axis=0
            )
        except Exception as e:
            print(f"[Sampler] lowpass_filter unavailable or failed ({e}) — skipping")

    # Use FlashSR's native chunk size (5.12s) unless overridden by user
    # We honour chunk_size and overlap from the node settings
    chunk_s = int(chunk_size * 48000) if chunk_size > 0 else FLASHSR_CHUNK_SAMPLES
    overlap_s = int(overlap * 48000) if overlap > 0 else FLASHSR_OVERLAP_SAMPLES

    channels = [waveform_48k[i] for i in range(waveform_48k.shape[0])]
    processed = []
    for channel in channels:
        _check_interrupted()
        # Call FlashSR — adjust this to the actual API from Task 1
        out = run_inference(
            channel,
            model_obj["ldm"],
            model_obj["vocoder"],
            model_obj["vae"],
            chunk_samples=chunk_s,
            overlap_samples=overlap_s,
        )
        processed.append(out)

    result = np.stack(processed, axis=0)

    # Resample to requested output SR if not 48kHz
    if output_sr != 48000:
        result = _resample(result, 48000, output_sr)

    return result


# ---------------------------------------------------------------------------
# Sampler node
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
                # --- AudioSR only ---
                "ddim_steps": ("INT", {
                    "default": 50, "min": 10, "max": 500, "step": 1,
                    "tooltip": "[AudioSR only] Denoising steps — higher = better quality, slower",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "[AudioSR only] Classifier-free guidance scale",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "[AudioSR only] Random seed (0 = random)",
                }),
                # --- FlashSR only ---
                "output_sr": (["48000", "44100", "96000"], {
                    "default": "48000",
                    "tooltip": "[FlashSR only] Target output sample rate",
                }),
                "lowpass_input": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[FlashSR only] Apply lowpass filter before inference",
                }),
                # --- Both ---
                "chunk_size": ("FLOAT", {
                    "default": 10.24, "min": 2.56, "max": 30.0, "step": 0.01,
                    "tooltip": "Chunk duration in seconds",
                }),
                "overlap": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Overlap between chunks in seconds",
                }),
                "attention_backend": (["sdpa", "sageattn", "eager"], {
                    "default": "sdpa",
                    "tooltip": "Attention backend (AudioSR). Ignored for FlashSR.",
                }),
                "unload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload model from VRAM after generation",
                }),
                "show_spectrogram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Output before/after spectrogram comparison image",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "IMAGE")
    RETURN_NAMES = ("audio", "spectrogram")
    FUNCTION = "run"
    CATEGORY = "audio"

    def run(
        self,
        audio,
        model: dict,
        ddim_steps: int = 50,
        guidance_scale: float = 3.5,
        seed: int = 0,
        output_sr: str = "48000",
        lowpass_input: bool = False,
        chunk_size: float = 10.24,
        overlap: float = 0.5,
        attention_backend: str = "sdpa",
        unload_model: bool = False,
        show_spectrogram: bool = True,
    ):
        # Use relative import when running as a package (ComfyUI), bare import otherwise (tests)
        try:
            from .vasr_node import generate_spectrogram_comparison
        except ImportError:
            from vasr_node import generate_spectrogram_comparison

        waveform, sr = _normalize_audio_input(audio)
        original_waveform = waveform.copy()
        original_sr = sr

        model_type = model["type"]
        models = model["models"]
        device = model["device"]
        dtype_str = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }.get(model["dtype"], "fp32")

        print(f"[Sampler] Running {model_type} on {waveform.shape[1] / sr:.2f}s audio ({sr}Hz)")

        with torch.no_grad():
            if model_type == "flashsr":
                result = _run_flashsr(
                    waveform, sr, models,
                    output_sr=int(output_sr),
                    lowpass_input=lowpass_input,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                out_sr = int(output_sr)
            else:
                result = _run_vasr(
                    waveform, sr, models,
                    ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    attention_backend=attention_backend,
                    dtype=dtype_str,
                )
                out_sr = 48000

        print(f"[Sampler] Done. Output: {result.shape[1] / out_sr:.2f}s at {out_sr}Hz")

        if unload_model:
            _evict_from_cache(model["cache_key"])

        # Spectrogram
        spec_tensor = torch.zeros((1, 256, 256, 3))
        if show_spectrogram:
            img = generate_spectrogram_comparison(original_waveform, original_sr, result, out_sr)
            if img is not None:
                import numpy as np
                from PIL import Image
                img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                spec_tensor = torch.from_numpy(arr).unsqueeze(0)
                spec_tensor = torch.clamp(spec_tensor, 0.0, 1.0)

        audio_out = _make_audio_output(result, out_sr)
        return (audio_out, spec_tensor)


def _evict_from_cache(cache_key: str):
    """Remove a model from the loader's cache and free VRAM."""
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
```

**Step 4: Run tests**

```bash
cd /media/p5/ComfyUI-Flash-AudioSR
python -m pytest tests/test_sampler_node.py -v
```

Expected: all 7 tests pass.

**Step 5: Commit**

```bash
git add sampler_node.py tests/test_sampler_node.py
git commit -m "feat: add unified AudioSRSampler node"
```

---

## Task 4: Update `__init__.py`

**Files:**
- Modify: `__init__.py`

### Step 1: Update `__init__.py`

Replace the current content of `__init__.py` with:

```python
from .vasr_node import NODE_CLASS_MAPPINGS as _vasr_mappings
from .vasr_node import NODE_DISPLAY_NAME_MAPPINGS as _vasr_display

try:
    from .model_loader import AudioSRModelLoader
    from .sampler_node import AudioSRSampler
    _new_nodes = {
        "AudioSRModelLoader": AudioSRModelLoader,
        "AudioSRSampler": AudioSRSampler,
    }
    _new_display = {
        "AudioSRModelLoader": "AudioSR Model Loader",
        "AudioSRSampler": "AudioSR Sampler",
    }
except Exception as e:
    print(f"[ComfyUI-Flash-AudioSR] Failed to load new nodes: {e}")
    _new_nodes = {}
    _new_display = {}

NODE_CLASS_MAPPINGS = {**_vasr_mappings, **_new_nodes}
NODE_DISPLAY_NAME_MAPPINGS = {**_vasr_display, **_new_display}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

### Step 2: Verify all nodes register correctly

```bash
cd /media/p5/ComfyUI-Flash-AudioSR
python -c "
from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
print('Registered nodes:')
for k, v in NODE_CLASS_MAPPINGS.items():
    print(f'  {k}: {v.__name__}')
"
```

Expected output:
```
Registered nodes:
  AudioSR: VASRNode
  AudioSRModelLoader: AudioSRModelLoader
  AudioSRSampler: AudioSRSampler
```

### Step 3: Commit

```bash
git add __init__.py
git commit -m "feat: register AudioSRModelLoader and AudioSRSampler nodes"
```

---

## Task 5: Adjust FlashSR inference API calls

After Task 1 reveals the actual FlashSR_Inference API, update the placeholder imports and calls in `sampler_node.py`:

**Files:**
- Modify: `sampler_node.py` — `_run_flashsr()` and `_load_flashsr_model()` in `model_loader.py`

### Step 1: Review actual API

Read the main inference file found in Task 1 and note:
- The actual class/function names for loading the 3 models
- The actual inference call signature (input format, output format)

### Step 2: Update `_load_flashsr_model` in `model_loader.py`

Replace the placeholder `from flashsr import load_student_ldm, load_vocoder, load_vae` with the actual loading code.

### Step 3: Update `_run_flashsr` in `sampler_node.py`

Replace the placeholder `from flashsr import run_inference` and its call with the actual inference code.

### Step 4: Run all tests

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (unit tests don't exercise FlashSR imports directly).

### Step 5: Smoke test with real audio (if GPU available)

```bash
python -c "
import torch
from model_loader import AudioSRModelLoader
from sampler_node import AudioSRSampler, _make_audio_output
import numpy as np

# Create 2s test audio at 22050Hz
waveform = np.random.randn(1, 44100).astype(np.float32) * 0.1
audio_in = {'waveform': torch.from_numpy(waveform).unsqueeze(0), 'sample_rate': 22050}

loader = AudioSRModelLoader()
model_out = loader.load_model('FlashSR', 'auto-download', 'cuda', 'fp32', True)

sampler = AudioSRSampler()
audio_out, spec = sampler.run(audio_in, model_out[0], show_spectrogram=False)
print('Output SR:', audio_out['sample_rate'])
print('Output shape:', audio_out['waveform'].shape)
print('OK')
"
```

### Step 6: Commit

```bash
git add model_loader.py sampler_node.py
git commit -m "feat: wire up actual FlashSR inference API"
```

---

## Task 6: Update `pyproject.toml` and `README.md`

**Files:**
- Modify: `pyproject.toml` — bump version to `1.2.0`
- Modify: `README.md` — document new nodes

### Step 1: Bump version

In `pyproject.toml`, update the version field to `1.2.0`.

### Step 2: Update README

Add a section describing:
- The two new nodes (`AudioSRModelLoader`, `AudioSRSampler`)
- The recommended workflow: `LoadAudio → AudioSRModelLoader → AudioSRSampler → PreviewAudio`
- That the original `AudioSR` node still works for backward compatibility
- Where models are stored (`ComfyUI/models/AudioSR/` and `ComfyUI/models/flashsr/`)

### Step 3: Commit

```bash
git add pyproject.toml README.md
git commit -m "docs: document new AudioSRModelLoader and AudioSRSampler nodes, bump to v1.2.0"
```
