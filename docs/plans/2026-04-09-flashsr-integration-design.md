# FlashSR Integration Design

**Date:** 2026-04-09  
**Status:** Approved

## Goal

Add FlashSR audio super-resolution support to ComfyUI-Flash-AudioSR alongside the existing AudioSR (VASR) model, using a clean two-node architecture (model loader + unified sampler) with auto-download for both models.

## Decisions

- **Approach:** Separate files (model loader + sampler), existing `vasr_node.py` kept for backward compat
- **FlashSR dependency:** Bundled as a minimal copy in `flashsr/` subfolder (mirrors how `versatile_audio_super_resolution/` is bundled)
- **Auto-download:** Both AudioSR and FlashSR, via `huggingface_hub`

## File Structure

```
flashsr/                        # bundled FlashSR inference code (minimal copy)
model_loader.py                 # AudioSRModelLoader node
sampler_node.py                 # AudioSRSampler node
vasr_node.py                    # kept as-is (backward compat)
__init__.py                     # updated to register all nodes
docs/plans/                     # design documents
```

## Custom Type: `AUDIOSR_MODEL`

A dict passed from model loader to sampler:

```python
{
    "type": "vasr" | "flashsr",
    "models": { ... },       # loaded model object(s)
    "device": "cuda" | "cpu",
    "dtype": torch.dtype,
    "cache_key": str,        # for cache invalidation (path + dtype + device)
}
```

For FlashSR, `models` contains: `{ "ldm": ..., "vocoder": ..., "vae": ... }` (the 3 checkpoints).  
For AudioSR, `models` contains: `{ "latent_diffusion": ... }`.

## Node: `AudioSRModelLoader`

**Category:** `audio`  
**Output:** `AUDIOSR_MODEL`

### Inputs

| Name | Type | Details |
|---|---|---|
| `model_type` | combo | `AudioSR`, `FlashSR` |
| `checkpoint` | combo | scans model dir; includes "auto-download" option |
| `device` | combo | `cuda`, `cpu` |
| `dtype` | combo | `fp32`, `fp16`, `bf16` |
| `auto_download` | boolean | download if not present (default: True) |

### Behavior

- Scans `ComfyUI/models/AudioSR/` for VASR checkpoints
- Scans `ComfyUI/models/flashsr/` for FlashSR weights
- If `auto_download` is True and files are missing, downloads via `huggingface_hub`:
  - AudioSR: `haoheliu/audiosr` → `ComfyUI/models/AudioSR/`
  - FlashSR: `jakeoneijk/FlashSR_weights` (3 files: `student_ldm.pth`, `sr_vocoder.pth`, `vae.pth`) → `ComfyUI/models/flashsr/`
- Caches loaded model in memory; invalidates cache on device/dtype/checkpoint change

## Node: `AudioSRSampler`

**Category:** `audio`  
**Inputs:** `AUDIO` + `AUDIOSR_MODEL` + sampler settings  
**Outputs:** `AUDIO`, `IMAGE` (spectrogram)

### Inputs

| Name | Type | Used by | Details |
|---|---|---|---|
| `audio` | AUDIO | Both | Input audio |
| `model` | AUDIOSR_MODEL | Both | From model loader |
| `chunk_size` | FLOAT | Both | Chunk duration in seconds |
| `overlap` | FLOAT | Both | Overlap between chunks |
| `attention_backend` | combo | Both | `sdpa`, `sageattn`, `eager` |
| `unload_model` | boolean | Both | Free VRAM after run |
| `show_spectrogram` | boolean | Both | Before/after spectrogram output |
| `ddim_steps` | INT | AudioSR only | Denoising steps (ignored for FlashSR) |
| `guidance_scale` | FLOAT | AudioSR only | CFG scale (ignored for FlashSR) |
| `seed` | INT | AudioSR only | Random seed (ignored for FlashSR) |
| `output_sr` | combo | FlashSR only | `44100`, `48000`, `96000` (ignored for AudioSR) |
| `lowpass_input` | boolean | FlashSR only | Pre-inference lowpass filter (ignored for AudioSR) |

### Behavior

- Reads `model["type"]` to dispatch to AudioSR or FlashSR inference path
- Settings not applicable to the loaded model are silently ignored
- Spectrogram generation reused from existing logic
- ComfyUI interrupt checking and progress updates on both paths

## Backward Compatibility

`vasr_node.py` and the original `VASRNode` remain registered and functional. Users with existing workflows are unaffected.
