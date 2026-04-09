# FlashSR Inference API Reference

Bundled from https://github.com/jakeoneijk/FlashSR_Inference  
Paper: https://arxiv.org/abs/2501.10807

---

## Overview

FlashSR is a one-step audio super-resolution model built on diffusion distillation. It upsamples audio to 48 kHz. The architecture consists of three components loaded separately:

| Component | Class | Weight file |
|---|---|---|
| Student LDM (denoiser) | `FlashSR` (inherits `DDPM`) | `student_ldm.pth` |
| SR Vocoder (BigVGAN-based) | `SRVocoder` | `sr_vocoder.pth` |
| VAE (autoencoder + HiFi-GAN) | `VAEWrapper` → `AutoencoderKL` | `vae.pth` |

---

## Weight Files

All weights should be downloaded from HuggingFace: `jakeoneijk/FlashSR_weights`

| Filename | Size/role |
|---|---|
| `student_ldm.pth` | Distilled LDM denoiser (UNet + full DDPM state dict) |
| `sr_vocoder.pth` | SRVocoder (BigVGAN-based mel→waveform vocoder with audio embedding) |
| `vae.pth` | AutoencoderKL weights (mel spectrogram encoder/decoder + HiFi-GAN vocoder) |

---

## Python Imports

```python
import sys
import os

# Add flashsr/ to sys.path so that FlashSR.* and TorchJaekwon.* resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flashsr'))

from FlashSR.FlashSR import FlashSR
from TorchJaekwon.Util.UtilAudio import UtilAudio
from TorchJaekwon.Util.UtilData import UtilData
```

---

## Loading the Model

All three components are loaded inside the `FlashSR.__init__()` constructor — no separate loading calls are needed.

```python
import torch

student_ldm_ckpt_path: str = '/path/to/ModelWeights/student_ldm.pth'
sr_vocoder_ckpt_path:  str = '/path/to/ModelWeights/sr_vocoder.pth'
vae_ckpt_path:         str = '/path/to/ModelWeights/vae.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FlashSR(
    student_ldm_ckpt_path,   # positional arg 1: DDPM student state dict
    sr_vocoder_ckpt_path,    # positional arg 2: SRVocoder state dict
    vae_ckpt_path,           # positional arg 3: AutoencoderKL state dict (mapped via VAEWrapper)
    model_output_type='v_prediction',   # default, do not change
    beta_schedule_type='cosine',        # default, do not change
)
model = model.to(device)
```

### What happens internally

1. `FlashSR.__init__` calls `super().__init__(model=AudioSRUnet(), ...)` to build the denoiser UNet.
2. `torch.load(student_ldm_ckpt_path)` → `self.load_state_dict(...)` loads the full DDPM state.
3. `VAEWrapper(vae_ckpt_path)` loads `AutoencoderKL` with weights; config is read from `FlashSR/AudioSR/args/model_argument.yaml`.
4. `SRVocoder()` is constructed with hardcoded defaults; `torch.load(sr_vocoder_ckpt_path)` → `self.sr_vocoder.load_state_dict(...)`.

---

## Inference Call

```python
# Input: [channel, time] float32 tensor — resampled to 48000 Hz
# IMPORTANT: The model processes exactly 245760 samples (5.12 seconds at 48 kHz) per call.
# For longer audio, split into 245760-sample chunks and concatenate outputs.

audio, sr = UtilAudio.read(audio_path, sample_rate=48000)
# audio.shape: [channels, time]  e.g. [2, 245760] for stereo

audio = UtilData.fix_length(audio, 245760)   # pad/trim to exactly 245760 samples
audio = audio.to(device)

pred_hr_audio = model(
    audio,                  # torch.Tensor [channel, time] — batch dimension NOT used here
    num_steps=1,            # number of diffusion steps; 1 is the distilled default (fast)
    lowpass_input=False,    # if True, apply Chebyshev lowpass before SR (helps for true LR audio)
    lowpass_cutoff_freq=None,  # if None and lowpass_input=True, auto-detected from STFT energy
)
# pred_hr_audio.shape: [channel, time] — 48 kHz high-resolution audio
```

### Batch inference

The forward pass works with a batch dimension too (the internal code comments show `[batch, time]`). Pass a 2D tensor:

```python
# batch of mono chunks:
audio_batch = audio.unsqueeze(0)   # [1, 245760]
pred = model(audio_batch, num_steps=1, lowpass_input=False)
# pred.shape: [1, 245760]
```

---

## Chunk Handling (for audio longer than 245760 samples)

The model has no built-in chunking. The caller must split the audio, process each chunk, and reassemble:

```python
CHUNK_SIZE = 245760   # 5.12 s at 48000 Hz

def run_flashsr_chunked(model, audio, device):
    # audio: [channels, total_time]
    channels, total = audio.shape
    output_chunks = []
    for start in range(0, total, CHUNK_SIZE):
        chunk = audio[:, start:start + CHUNK_SIZE]
        chunk = UtilData.fix_length(chunk, CHUNK_SIZE).to(device)
        with torch.no_grad():
            out = model(chunk, num_steps=1, lowpass_input=False)
        output_chunks.append(out.cpu())
    return torch.cat(output_chunks, dim=-1)[:, :total]
```

---

## Internal Forward Pass (FlashSR.forward summary)

1. Optional lowpass filter on LR input (Chebyshev order-8 at auto-detected cutoff).
2. `DiffusersWrapper.infer(ddpm_module=self, diffusers_scheduler_class=DPMSolverMultistepScheduler, x_shape=None, cond=lr_audio, num_steps=num_steps, device=...)` runs the diffusion reverse process.
3. Internally, `preprocess` calls `self.vae.encode_to_z(lr_audio)` to get the latent condition `z`.
4. UNet denoiser `AudioSRUnet` iterates over scheduler timesteps.
5. `postprocess` decodes via `self.vae.z_to_mel(x)` then `self.sr_vocoder(mel_spec, norm_wav)` to produce waveform.
6. Output is denormalized and returned as a waveform tensor.

---

## Audio I/O Utilities (from TorchJaekwon)

```python
from TorchJaekwon.Util.UtilAudio import UtilAudio

# Read (returns [channels, samples], sr)
audio, sr = UtilAudio.read(path, sample_rate=48000)

# Write
UtilAudio.write(output_path, pred_hr_audio, 48000)
```

---

## Output Format

- Shape: `[channels, time]` — same channel count and sample count as input
- Sample rate: 48000 Hz
- dtype: float32
- Range: approximately [-1, 1] (denormalized waveform)
