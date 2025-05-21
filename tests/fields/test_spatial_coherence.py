import pytest
import torch
from matplotlib.figure import Figure

from torchoptics import Field, SpatialCoherence
from torchoptics.elements import Modulator
from torchoptics.functional import outer2d


def make_spatial_coherence_fixture():
    shape = (20, 21)
    wavelength = 795e-9
    z = 0
    spacing = 9.2e-6
    offset = (-102e-6, 83e-6)
    input_field = torch.rand(shape, dtype=torch.double) * torch.exp(
        2j * torch.pi * torch.rand(shape, dtype=torch.double),
    )
    input_spatial_coherence = outer2d(input_field, input_field)
    field = Field(input_field, wavelength, z, spacing, offset)
    spatial_coherence = SpatialCoherence(input_spatial_coherence, wavelength, z, spacing, offset)
    return (
        field,
        spatial_coherence,
        input_field,
        input_spatial_coherence,
        shape,
        wavelength,
        z,
        spacing,
        offset,
    )


def test_spatial_coherence_incorrect_shape() -> None:
    _, _, _, _, _, wavelength, z, spacing, offset = make_spatial_coherence_fixture()
    with pytest.raises(ValueError):
        SpatialCoherence(
            torch.ones(2, 3),
            wavelength=wavelength,
            z=z,
            spacing=spacing,
            offset=offset,
        )
    with pytest.raises(ValueError):
        SpatialCoherence(
            torch.ones(2, 3, 2, 5),
            wavelength=wavelength,
            z=z,
            spacing=spacing,
            offset=offset,
        ).intensity()


def test_spatial_coherence_intensity_equal_field_coherent() -> None:
    field, spatial_coherence, *_ = make_spatial_coherence_fixture()
    assert torch.allclose(field.intensity(), spatial_coherence.intensity())


def test_spatial_coherence_modulation_intensity() -> None:
    field, spatial_coherence, _, _, shape, _, z, spacing, offset = make_spatial_coherence_fixture()
    modulator = Modulator(
        torch.rand(shape) * torch.exp(2j * torch.pi * torch.rand(shape)),
        z,
        spacing,
        offset,
    )
    modulated_field = modulator.forward(field)
    modulated_spatial_coherence = modulator.forward(spatial_coherence)
    assert torch.allclose(modulated_field.intensity(), modulated_spatial_coherence.intensity())


def test_spatial_coherence_propagation_intensity() -> None:
    field, spatial_coherence, *_ = make_spatial_coherence_fixture()
    prop_shape = (23, 24)
    prop_z = 0.1
    prop_spacing = 9.0e-6
    prop_offset = (-11e-6, 50e-6)
    prop_field = field.propagate(prop_shape, prop_z, prop_spacing, prop_offset)
    prop_spatial_coherence = spatial_coherence.propagate(prop_shape, prop_z, prop_spacing, prop_offset)
    assert torch.allclose(prop_field.intensity(), prop_spatial_coherence.intensity())
    assert prop_field.is_same_geometry(prop_spatial_coherence)


def test_spatial_coherence_normalization_coherent() -> None:
    field, spatial_coherence, *_ = make_spatial_coherence_fixture()
    normalized_power = 2.53
    field_norm = field.normalize(normalized_power)
    spatial_coherence_norm = spatial_coherence.normalize(normalized_power)
    assert torch.allclose(field_norm.intensity(), spatial_coherence_norm.intensity())
    assert torch.allclose(spatial_coherence_norm.power(), torch.tensor(normalized_power, dtype=torch.double))


def test_spatial_coherence_visualization() -> None:
    _, spatial_coherence, *_ = make_spatial_coherence_fixture()
    fig = spatial_coherence.visualize(return_fig=True, show=False)
    assert isinstance(fig, Figure)


def test_spatial_coherence_raise_error() -> None:
    shape = (20, 21)
    wavelength = 795e-9
    z = 0
    spacing = 9.2e-6
    offset = (-102e-6, 83e-6)
    input_field = torch.rand(shape, dtype=torch.double) * torch.exp(
        2j * torch.pi * torch.rand(shape, dtype=torch.double),
    )
    input_spatial_coherence = outer2d(input_field, input_field)
    input_spatial_coherence[0, 3] = input_spatial_coherence[3, 0] + 2  # Make non-Hermitian
    spatial_coherence = SpatialCoherence(input_spatial_coherence, wavelength, z, spacing, offset)
    with pytest.raises(ValueError):
        spatial_coherence.intensity()


def test_spatial_coherence_inner_outer() -> None:
    spatial_coherence1 = SpatialCoherence(torch.ones(10, 10, 10, 10), spacing=1, wavelength=1)
    spatial_coherence2 = SpatialCoherence(torch.ones(10, 10, 10, 10), spacing=1, wavelength=1)
    with pytest.raises(TypeError):
        spatial_coherence1.inner(spatial_coherence2)
    with pytest.raises(TypeError):
        spatial_coherence1.outer(spatial_coherence2)
