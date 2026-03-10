import torch

from torchoptics import Field, SpatialCoherence
from torchoptics.profiles import gaussian, gaussian_schell_model


def test_gaussian_schell_model_shape():
    shape = (10, 15)
    waist_radius = 50e-6
    coherence_width = torch.inf
    spacing = 10e-6
    coherence_data = gaussian_schell_model(
        shape=shape,
        waist_radius=waist_radius,
        coherence_width=coherence_width,
        spacing=spacing,
    )
    assert coherence_data.shape == (10, 15, 10, 15)
    assert coherence_data.dtype == torch.double


def test_gaussian_schell_model_identical_with_gaussian():
    shape = (10, 15)
    waist_radius = 50e-6
    coherence_width = torch.inf
    spacing = 10e-6
    wavelength = 700e-9
    coherence_data = gaussian_schell_model(
        shape=shape,
        waist_radius=waist_radius,
        coherence_width=coherence_width,
        spacing=spacing,
    )
    gaussian_data = gaussian(
        shape=shape,
        waist_radius=waist_radius,
        wavelength=1,
        spacing=spacing,
    )
    field = Field(gaussian_data, spacing=spacing, wavelength=wavelength)
    spatial_coherence = SpatialCoherence(coherence_data, spacing=spacing, wavelength=wavelength)
    assert torch.allclose(field.intensity(), spatial_coherence.intensity())
    assert torch.allclose(
        field.propagate_to_z(0.2).intensity(), spatial_coherence.propagate_to_z(0.2).intensity()
    )
    assert coherence_data.dtype == torch.double
    assert gaussian_data.dtype == torch.cdouble


def test_gaussian_schell_model_incoherent():
    shape = (10, 15)
    waist_radius = 50e-6
    spacing = 10e-6
    incoherent_data = gaussian_schell_model(
        shape=shape,
        waist_radius=waist_radius,
        coherence_width=0,
        spacing=spacing,
    )
    incoherent_data = incoherent_data.view(shape[0] * shape[1], -1)
    incoherent_data[torch.eye(shape[0] * shape[1], dtype=torch.bool)] = 0
    assert torch.all(incoherent_data == 0)
