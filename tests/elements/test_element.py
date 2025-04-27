import pytest
import torch

from torchoptics import Field
from torchoptics.elements import Element


def test_element():
    shape = (32, 32)
    z = 0
    spacing = 1
    element = Element(shape, z, spacing=spacing)
    with pytest.raises(TypeError):
        element.validate_field("not a field")
    field = Field(torch.ones(3, *shape), wavelength=700e-9, spacing=spacing, z=2)
    with pytest.raises(ValueError):
        element.validate_field(field)
