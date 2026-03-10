import torch

from torchoptics import Field
from torchoptics.elements import CylindricalLens


def test_cylindrical_lens():
    shape = (64, 64)
    focal_length = 50.0
    wavelength = 500e-9
    spacing = 1e-5
    lens = CylindricalLens(shape, focal_length, 0, spacing=spacing)
    assert lens.shape == shape
    assert lens.focal_length == focal_length
    assert lens.modulation_profile(wavelength).dtype == torch.cdouble
    field = Field(torch.ones(3, *shape), wavelength=wavelength, spacing=spacing)
    output_field = lens(field)
    assert isinstance(output_field, Field)
