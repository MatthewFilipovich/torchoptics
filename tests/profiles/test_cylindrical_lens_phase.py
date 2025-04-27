import torch

from torchoptics import PlanarGrid
from torchoptics.profiles import cylindrical_lens_phase


def test_cylindrical_lens_phase_shape_and_dtype():
    shape = (100, 100)
    focal_length = 50.0
    theta = torch.pi / 4
    wavelength = 0.5
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    phase_profile = cylindrical_lens_phase(
        shape=shape,
        focal_length=focal_length,
        theta=theta,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    assert phase_profile.shape == shape
    assert phase_profile.dtype == torch.double


def test_cylindrical_lens_phase_zero_theta():
    shape = (100, 100)
    focal_length = 50.0
    wavelength = 0.5
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    phase_profile_zero_theta = cylindrical_lens_phase(
        shape=shape,
        focal_length=focal_length,
        theta=0.0,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    x, _ = planar_grid.meshgrid()
    expected_phase = -torch.pi / (wavelength * focal_length) * x**2
    assert torch.allclose(phase_profile_zero_theta, expected_phase, atol=1e-5)


def test_cylindrical_lens_phase_pi_over_2_theta():
    shape = (100, 100)
    focal_length = 50.0
    wavelength = 0.5
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    phase_profile_pi_over_2_theta = cylindrical_lens_phase(
        shape=shape,
        focal_length=focal_length,
        theta=torch.pi / 2,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    _, y = planar_grid.meshgrid()
    expected_phase = -torch.pi / (wavelength * focal_length) * y**2
    assert torch.allclose(phase_profile_pi_over_2_theta, expected_phase, atol=1e-5)
