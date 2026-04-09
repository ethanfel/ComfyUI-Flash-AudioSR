"""Tests for AudioSRSampler node."""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampler_node import AudioSRSampler, _normalize_audio_input, _make_audio_output


def make_audio(sr=22050, duration=1.0, channels=1):
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
    audio = make_audio(sr=44100, duration=0.5, channels=1)
    waveform, sr = _normalize_audio_input(audio)
    assert sr == 44100
    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 2


def test_normalize_audio_stereo():
    audio = make_audio(sr=48000, duration=0.5, channels=2)
    waveform, sr = _normalize_audio_input(audio)
    assert waveform.shape[0] == 2


def test_normalize_audio_tuple():
    samples = int(22050 * 0.5)
    waveform_t = torch.randn(1, 1, samples)
    audio = (waveform_t, 22050)
    waveform, sr = _normalize_audio_input(audio)
    assert sr == 22050
    assert isinstance(waveform, np.ndarray)


def test_make_audio_output():
    waveform = np.random.randn(2, 48000).astype(np.float32)
    result = _make_audio_output(waveform, 48000)
    assert "waveform" in result
    assert "sample_rate" in result
    assert result["sample_rate"] == 48000
    assert result["waveform"].shape == (1, 2, 48000)
