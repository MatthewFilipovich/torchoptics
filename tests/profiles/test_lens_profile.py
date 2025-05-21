import torch

from torchoptics.profiles import lens_phase


def test_lens_profile_shape_and_dtype() -> None:
    shape = (100, 100)
    focal_length = 50.0
    wavelength = 0.5
    spacing = (0.1, 0.1)
    offset = (0.0, 0.0)
    phase_profile = lens_phase(
        shape=shape,
        focal_length=focal_length,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    assert phase_profile.shape == shape
    assert phase_profile.dtype == torch.double
