"""Tests for AudioSRModelLoader node."""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import get_model_dir, build_cache_key, AudioSRModelLoader


def test_get_model_dir_vasr():
    result = get_model_dir("vasr")
    assert "AudioSR" in result

def test_get_model_dir_flashsr():
    result = get_model_dir("flashsr")
    assert "flashsr" in result

def test_build_cache_key():
    assert build_cache_key("vasr", "model.safetensors", "cuda", "fp32") == "vasr:model.safetensors:cuda:fp32"

def test_build_cache_key_flashsr():
    assert build_cache_key("flashsr", "auto", "cpu", "fp16") == "flashsr:auto:cpu:fp16"

def test_input_types_structure():
    inputs = AudioSRModelLoader.INPUT_TYPES()
    assert "required" in inputs
    required = inputs["required"]
    assert "model_type" in required
    assert "device" in required
    assert "dtype" in required
    assert "auto_download" in required

def test_return_types():
    assert AudioSRModelLoader.RETURN_TYPES == ("AUDIOSR_MODEL",)
