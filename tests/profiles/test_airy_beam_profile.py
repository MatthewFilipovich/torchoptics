import pytest
import torch

from torchoptics.profiles import airy_beam
from torchoptics.profiles._profile_meshgrid import profile_meshgrid


def test_airy_beam_shape_and_formula():
    shape = (64, 64)
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    scale = 1.5
    truncation = 0.2

    profile = airy_beam(
        shape=shape,
        scale=scale,
        truncation=truncation,
        spacing=spacing,
        offset=offset,
    )

    x, y = profile_meshgrid(shape, spacing, offset)
    expected = torch.special.airy_ai(x / scale) * torch.exp(truncation * x / scale)
    expected = expected * torch.special.airy_ai(y / scale) * torch.exp(truncation * y / scale)

    assert profile.shape == shape
    assert torch.allclose(profile, expected)


def test_airy_beam_defaults_to_profile_meshgrid_coordinates():
    shape = (32, 32)
    scale = 1.0
    truncation = 0.1
    spacing = (0.1, 0.1)

    profile = airy_beam(shape=shape, scale=scale, truncation=truncation, spacing=spacing)
    x, y = profile_meshgrid(shape, spacing, None)

    expected = torch.special.airy_ai(x / scale) * torch.exp(truncation * x / scale)
    expected = expected * torch.special.airy_ai(y / scale) * torch.exp(truncation * y / scale)

    assert torch.allclose(profile, expected)


def test_airy_beam_invalid_parameters():
    shape = (16, 16)

    with pytest.raises(ValueError, match="scale"):
        airy_beam(shape=shape, scale=0.0, truncation=0.1)

    with pytest.raises(ValueError, match="truncation"):
        airy_beam(shape=shape, scale=1.0, truncation=-0.1)
