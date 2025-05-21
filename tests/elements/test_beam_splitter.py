import torch

import torchoptics
from torchoptics.elements import BeamSplitter


def test_beam_splitter() -> None:
    shape = 64
    z = 0
    field = torchoptics.Field(torch.ones(3, shape, shape), wavelength=700e-9, spacing=1e-5)
    bs = BeamSplitter(shape, theta=torch.pi / 4, phi_0=0, phi_r=0, phi_t=0, z=z, spacing=1e-5)
    assert bs.shape == (shape, shape)

    # Single field
    bs_field0, bs_field1 = bs.forward(field)
    assert isinstance(bs_field0, torchoptics.Field)
    assert isinstance(bs_field1, torchoptics.Field)
    assert torch.allclose(bs_field0.intensity(), 0.5 * field.intensity())
    assert torch.allclose(bs_field1.intensity(), 0.5 * field.intensity())

    # Two fields
    bs_field0, bs_field1 = bs.forward(field, field)
    assert isinstance(bs_field0, torchoptics.Field)
    assert isinstance(bs_field1, torchoptics.Field)
    assert torch.allclose(bs_field0.intensity(), 2 * field.intensity())
    assert torch.allclose(bs_field1.intensity(), 0 * field.intensity())

    polarized_field = torchoptics.Field(torch.ones(4, 3, shape, shape), wavelength=700e-9, spacing=1e-5)
    # Single polarized field
    bs_polarized_field0, bs_polarized_field1 = bs.forward(polarized_field)
    assert isinstance(bs_polarized_field0, torchoptics.Field)
    assert isinstance(bs_polarized_field1, torchoptics.Field)
    assert torch.allclose(bs_polarized_field0.intensity(), 0.5 * polarized_field.intensity())
    assert torch.allclose(bs_polarized_field1.intensity(), 0.5 * polarized_field.intensity())
    # Two polarized fields
    bs_polarized_field0, bs_polarized_field1 = bs.forward(polarized_field, polarized_field)
    assert isinstance(bs_polarized_field0, torchoptics.Field)
    assert isinstance(bs_polarized_field1, torchoptics.Field)
    assert torch.allclose(bs_polarized_field0.intensity(), 2 * polarized_field.intensity())
    assert torch.allclose(bs_polarized_field1.intensity(), 0 * polarized_field.intensity())
