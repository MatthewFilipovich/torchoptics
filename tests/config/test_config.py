import pytest
import torch

from torchoptics.config import (
    get_default_dtype,
    get_default_spacing,
    get_default_wavelength,
    set_default_dtype,
    set_default_spacing,
    set_default_wavelength,
)


def test_get_default_spacing_not_set():
    with pytest.raises(ValueError):
        get_default_spacing()


def test_set_and_get_default_spacing():
    spacing = (10e-6, 10e-6)
    set_default_spacing(spacing)
    result = get_default_spacing()
    assert torch.allclose(result, torch.tensor(spacing, dtype=result.dtype))


def test_get_default_wavelength_not_set():
    with pytest.raises(ValueError):
        get_default_wavelength()


def test_set_and_get_default_wavelength():
    wavelength = 700e-6
    set_default_wavelength(wavelength)
    result = get_default_wavelength()
    assert torch.allclose(result, torch.tensor(wavelength, dtype=result.dtype))


def test_default_dtype_defaults_to_float_for_tests():
    assert get_default_dtype() == torch.float32


def test_set_and_get_default_dtype():
    original_dtype = get_default_dtype()
    try:
        set_default_dtype(torch.float32)
        assert get_default_dtype() == torch.float32
    finally:
        set_default_dtype(original_dtype)


def test_set_default_dtype_validates_type():
    with pytest.raises(TypeError):
        set_default_dtype("float32")  # pyright: ignore[reportArgumentType]


def test_set_default_dtype_validates_supported_values():
    with pytest.raises(ValueError):
        set_default_dtype(torch.float16)
