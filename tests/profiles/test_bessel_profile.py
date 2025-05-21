import torch

from torchoptics.profiles import bessel


def test_bessel_shape_and_dtype() -> None:
    shape = (100, 100)
    cone_angle = torch.pi / 4
    wavelength = 0.5
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    profile = bessel(
        shape=shape,
        cone_angle=cone_angle,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    assert profile.shape == shape
    assert profile.dtype == torch.double


def test_bessel_values() -> None:
    shape = (100, 100)
    cone_angle = torch.pi / 4
    wavelength = 0.5
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    profile = bessel(
        shape=shape,
        cone_angle=cone_angle,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    assert torch.all(profile.abs() <= 1)
