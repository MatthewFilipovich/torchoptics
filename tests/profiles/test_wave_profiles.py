import torch

from torchoptics.profiles import plane_wave_phase, spherical_wave_phase


def test_plane_wave_shape_and_dtype():
    shape = (100, 100)
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    wavelength = 0.5
    z = 1.0
    theta = torch.pi / 4
    phi = torch.pi / 6
    wave = plane_wave_phase(
        shape=shape,
        theta=theta,
        phi=phi,
        z=z,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    assert wave.shape == shape
    assert wave.dtype == torch.double


def test_spherical_wave_shape_and_dtype():
    shape = (100, 100)
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    wavelength = 0.5
    z = 1.0
    wave = spherical_wave_phase(
        shape=shape,
        z=z,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    assert wave.shape == shape
    assert wave.dtype == torch.double
