import pytest
import torch
from matplotlib.figure import Figure

import torchoptics
from torchoptics import Field
from torchoptics.elements import PolarizedAmplitudeModulator, PolarizedModulator, PolarizedPhaseModulator


def test_polarized_modulator_initialization():
    polarized_modulation_profile = torch.rand((3, 3, 10, 12), dtype=torch.cdouble)
    z = 1.5
    polarized_field = Field(torch.ones(4, 3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    torchoptics.set_default_spacing(1)
    modulator = PolarizedModulator(polarized_modulation_profile, z)
    assert isinstance(modulator(polarized_field), Field)
    assert torch.equal(modulator.polarized_modulation_profile(), polarized_modulation_profile)


def test_polarized_phase_modulator_initialization_and_profile():
    phase_profile = torch.rand((3, 3, 10, 12))
    z = 1.5
    phase_modulator = PolarizedPhaseModulator(phase_profile, z)
    expected_profile = torch.exp(1j * phase_profile).to(dtype=torch.cdouble)
    assert torch.allclose(phase_modulator.polarized_modulation_profile(), expected_profile)
    polarized_field = Field(torch.ones(4, 3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    assert isinstance(phase_modulator(polarized_field), Field)


def test_polarized_amplitude_modulator_initialization_and_profile():
    amplitude_profile = torch.rand((3, 3, 10, 12))
    z = 1.5
    amplitude_modulator = PolarizedAmplitudeModulator(amplitude_profile, z)
    expected_profile = amplitude_profile.to(torch.cdouble)
    assert torch.allclose(amplitude_modulator.polarized_modulation_profile(), expected_profile)
    polarized_field = Field(torch.ones(4, 3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    assert isinstance(amplitude_modulator(polarized_field), Field)


def test_phase_modulation_profile_consistency():
    phase_profile = torch.rand((3, 3, 10, 12))
    z = 1.5
    phase_modulator = PolarizedPhaseModulator(phase_profile, z)
    modulator = PolarizedModulator(torch.exp(1j * phase_profile), z)
    assert torch.allclose(
        modulator.polarized_modulation_profile(), phase_modulator.polarized_modulation_profile()
    )


def test_amplitude_modulation_profile_consistency():
    amplitude_profile = torch.rand((3, 3, 10, 12))
    z = 1.5
    amplitude_modulator = PolarizedAmplitudeModulator(amplitude_profile, z)
    modulator = PolarizedModulator(amplitude_profile.to(torch.cdouble), z)
    assert torch.allclose(
        modulator.polarized_modulation_profile(), amplitude_modulator.polarized_modulation_profile()
    )


def test_error_on_invalid_tensor_input():
    z = 1.5
    with pytest.raises(TypeError):
        PolarizedModulator("not a tensor", z)  # type: ignore
    with pytest.raises(TypeError):
        PolarizedPhaseModulator("not a tensor", z)  # type: ignore
    with pytest.raises(TypeError):
        PolarizedAmplitudeModulator("not a tensor", z)  # type: ignore


def test_error_on_incorrect_dimensions():
    z = 1.5
    with pytest.raises(ValueError):
        PolarizedModulator(torch.rand((3, 3, 10)), z)
    with pytest.raises(ValueError):
        PolarizedModulator(torch.rand((3, 4, 10, 10)), z)


def test_visualization():
    polarized_modulation_profile = torch.rand((3, 3, 10, 12), dtype=torch.cdouble)
    z = 1.5
    modulator = PolarizedModulator(polarized_modulation_profile, z)
    fig = modulator.visualize(0, 0, show=False, return_fig=True)
    assert isinstance(fig, Figure)
