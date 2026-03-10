import pytest
import torch

from torchoptics.config import (
    get_default_spacing,
    get_default_wavelength,
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
    assert torch.allclose(result, torch.tensor(spacing, dtype=torch.double))


def test_get_default_wavelength_not_set():
    with pytest.raises(ValueError):
        get_default_wavelength()


def test_set_and_get_default_wavelength():
    wavelength = 700e-6
    set_default_wavelength(wavelength)
    result = get_default_wavelength()
    assert torch.allclose(result, torch.tensor(wavelength, dtype=torch.double))
