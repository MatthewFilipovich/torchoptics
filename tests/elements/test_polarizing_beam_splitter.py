import pytest
import torch

from torchoptics import Field
from torchoptics.elements import PolarizingBeamSplitter


def test_polarizing_beam_splitter():
    shape = 32
    z = 0
    field = Field(torch.ones(4, 3, shape, shape), wavelength=700e-9, spacing=1e-5)
    field.data[:, 2] = 0  # z polarization
    bs = PolarizingBeamSplitter(shape, z, spacing=1e-5)
    assert bs.shape == (shape, shape)

    bs_field0, bs_field1 = bs.forward(field)
    assert isinstance(bs_field0, Field)
    assert isinstance(bs_field1, Field)
    assert torch.allclose(bs_field0.data[:, 0], field.data[:, 0])
    assert torch.allclose(bs_field0.data[:, 1], 0 * field.data[:, 0])
    assert torch.allclose(bs_field1.data[:, 0], 0 * field.data[:, 0])
    assert torch.allclose(bs_field1.data[:, 1], field.data[:, 1])

    # Test with z-polarized field (should raise ValueError)
    field_z = Field(torch.ones(1, 2, shape, shape), wavelength=700e-9, spacing=1e-5)
    with pytest.raises(ValueError):
        bs.forward(field_z)
