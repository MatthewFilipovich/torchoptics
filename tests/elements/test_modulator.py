import matplotlib.pyplot as plt
import pytest
import torch

import torchoptics
from torchoptics import Field
from torchoptics.elements import AmplitudeModulator, Modulator, PhaseModulator, PolychromaticPhaseModulator


def test_modulator_initialization():
    complex_tensor = torch.rand((10, 12), dtype=torch.cdouble)
    z = 1.5
    field = Field(torch.ones(3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    torchoptics.set_default_spacing(1)
    modulator = Modulator(complex_tensor, z)
    assert isinstance(modulator(field), Field)
    assert torch.equal(modulator.modulation_profile(), complex_tensor)


def test_phase_modulator_initialization_and_profile():
    phase_profile = torch.rand((10, 12))
    z = 1.5
    phase_modulator = PhaseModulator(phase_profile, z)
    expected_profile = torch.exp(1j * phase_profile).cdouble()
    assert torch.allclose(phase_modulator.modulation_profile(), expected_profile)
    field = Field(torch.ones(3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    assert isinstance(phase_modulator(field), Field)


def test_amplitude_modulator_initialization_and_profile():
    amplitude_profile = torch.rand((10, 12))
    z = 1.5
    amplitude_modulator = AmplitudeModulator(amplitude_profile, z)
    expected_profile = amplitude_profile.cdouble()
    assert torch.allclose(amplitude_modulator.modulation_profile(), expected_profile)
    field = Field(torch.ones(3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    assert isinstance(amplitude_modulator(field), Field)


def test_phase_modulation_profile_consistency():
    phase_profile = torch.rand((10, 12))
    z = 1.5
    phase_modulator = PhaseModulator(phase_profile, z)
    modulator = Modulator(torch.exp(1j * phase_profile), z)
    assert torch.allclose(modulator.modulation_profile(), phase_modulator.modulation_profile())


def test_polychromatic_phase_modulator():
    optical_path_length = torch.rand((10, 12), dtype=torch.double)
    z = 1.5
    polychromatic_modulator = PolychromaticPhaseModulator(optical_path_length, z)
    wavelength = 700e-9
    expected_profile = torch.exp(2j * torch.pi / wavelength * optical_path_length)
    assert torch.allclose(polychromatic_modulator.modulation_profile(wavelength), expected_profile)
    field = Field(torch.ones(3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    assert isinstance(polychromatic_modulator(field), Field)


def test_amplitude_modulation_profile_consistency():
    amplitude_profile = torch.rand((10, 12))
    z = 1.5
    amplitude_modulator = AmplitudeModulator(amplitude_profile, z)
    modulator = Modulator(amplitude_profile.cdouble(), z)
    assert torch.allclose(modulator.modulation_profile(), amplitude_modulator.modulation_profile())


def test_error_on_invalid_tensor_input():
    z = 1.5
    with pytest.raises(TypeError):
        Modulator("not a tensor", z)
    with pytest.raises(TypeError):
        PhaseModulator("not a tensor", z)
    with pytest.raises(TypeError):
        AmplitudeModulator("not a tensor", z)


def test_error_on_incorrect_dimensions():
    z = 1.5
    invalid_tensor = torch.rand((10, 10, 10))
    with pytest.raises(ValueError):
        Modulator(invalid_tensor, z)
    with pytest.raises(ValueError):
        PhaseModulator(invalid_tensor, z)
    with pytest.raises(ValueError):
        AmplitudeModulator(invalid_tensor, z)


def test_visualization():
    complex_tensor = torch.rand((10, 12), dtype=torch.cdouble)
    z = 1.5
    modulator = Modulator(complex_tensor, z)
    fig = modulator.visualize(show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)


def test_polychromatic_visualization():
    optical_path_length = torch.rand((10, 12), dtype=torch.double)
    z = 1.5
    polychromatic_modulator = PolychromaticPhaseModulator(optical_path_length, z)
    fig = polychromatic_modulator.visualize(700e-9, show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)
