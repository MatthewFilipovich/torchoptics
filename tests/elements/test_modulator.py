import pytest
import torch
from matplotlib.figure import Figure

import torchoptics
from torchoptics import Field
from torchoptics.elements import AmplitudeModulator, Modulator, PhaseModulator, PolychromaticPhaseModulator


def test_modulator_initialization():
    complex_tensor = torch.rand((10, 12), dtype=torch.cfloat)
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
    expected_profile = torch.exp(1j * phase_profile).to(torch.cfloat)
    assert torch.allclose(phase_modulator.modulation_profile(), expected_profile)
    field = Field(torch.ones(3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    assert isinstance(phase_modulator(field), Field)


def test_amplitude_modulator_initialization_and_profile():
    amplitude_profile = torch.rand((10, 12))
    z = 1.5
    amplitude_modulator = AmplitudeModulator(amplitude_profile, z)
    expected_profile = amplitude_profile.to(torch.cfloat)
    assert torch.allclose(amplitude_modulator.modulation_profile(), expected_profile)
    field = Field(torch.ones(3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    assert isinstance(amplitude_modulator(field), Field)


def test_phase_modulation_profile_consistency():
    phase_profile = torch.rand((10, 12))
    z = 1.5
    phase_modulator = PhaseModulator(phase_profile, z)
    modulator = Modulator(torch.exp(1j * phase_profile), z)
    assert torch.allclose(modulator.modulation_profile(), phase_modulator.modulation_profile())


def test_polychromatic_phase_modulator_scalar_n():
    thickness = torch.rand((10, 12))
    n = 1.5
    z = 1.5
    modulator = PolychromaticPhaseModulator(thickness, n=n, z=z)
    wavelength = 700e-9
    expected = torch.exp(2j * torch.pi / torch.tensor(wavelength) * (n - 1) * thickness)
    assert torch.allclose(modulator.modulation_profile(wavelength), expected)
    field = Field(torch.ones(3, 10, 12), wavelength=700e-9, z=z, spacing=1)
    assert isinstance(modulator(field), Field)


def test_polychromatic_phase_modulator_callable_n():
    thickness = torch.rand((10, 12))
    z = 1.5

    def n_func(wl):
        return 1.5 + 0.01e-12 / wl**2

    modulator = PolychromaticPhaseModulator(thickness, n=n_func, z=z)
    wavelength = 700e-9
    n_val = n_func(wavelength)
    expected = torch.exp(2j * torch.pi / torch.tensor(wavelength) * (n_val - 1) * thickness)
    assert torch.allclose(modulator.modulation_profile(wavelength), expected)


def test_polychromatic_phase_modulator_callable_n_wavelength_dependence():
    """Verify that a callable n produces different profiles at different wavelengths."""
    thickness = torch.ones((10, 12))
    z = 1.5

    def n_func(wl):
        return 1.5 + 0.01e-12 / wl**2

    modulator = PolychromaticPhaseModulator(thickness, n=n_func, z=z)
    profile_400 = modulator.modulation_profile(400e-9)
    profile_700 = modulator.modulation_profile(700e-9)
    assert not torch.allclose(profile_400, profile_700)


def test_polychromatic_phase_modulator_n1():
    """With n=1, (n-1)=0, so modulation should be all ones."""
    thickness = torch.rand((10, 12))
    z = 1.5
    modulator = PolychromaticPhaseModulator(thickness, n=1, z=z)
    wavelength = 700e-9
    profile = modulator.modulation_profile(wavelength)
    assert torch.allclose(profile, torch.ones_like(profile))


def test_amplitude_modulation_profile_consistency():
    amplitude_profile = torch.rand((10, 12))
    z = 1.5
    amplitude_modulator = AmplitudeModulator(amplitude_profile, z)
    modulator = Modulator(amplitude_profile.to(torch.cfloat), z)
    assert torch.allclose(modulator.modulation_profile(), amplitude_modulator.modulation_profile())


def test_error_on_invalid_tensor_input():
    z = 1.5
    with pytest.raises(TypeError):
        Modulator("not a tensor", z)  # type: ignore
    with pytest.raises(TypeError):
        PhaseModulator("not a tensor", z)  # type: ignore
    with pytest.raises(TypeError):
        AmplitudeModulator("not a tensor", z)  # type: ignore
    with pytest.raises(TypeError):
        PolychromaticPhaseModulator("not a tensor", n=1.5, z=z)  # type: ignore


def test_error_on_incorrect_dimensions():
    z = 1.5
    invalid_tensor = torch.rand((10, 10, 10))
    with pytest.raises(ValueError):
        Modulator(invalid_tensor, z)
    with pytest.raises(ValueError):
        PhaseModulator(invalid_tensor, z)
    with pytest.raises(ValueError):
        AmplitudeModulator(invalid_tensor, z)
    with pytest.raises(ValueError):
        PolychromaticPhaseModulator(invalid_tensor, n=1.5, z=z)


def test_visualization():
    complex_tensor = torch.rand((10, 12), dtype=torch.cfloat)
    z = 1.5
    modulator = Modulator(complex_tensor, z)
    fig = modulator.visualize(show=False, return_fig=True)
    assert isinstance(fig, Figure)


def test_polychromatic_visualization():
    thickness = torch.rand((10, 12))
    z = 1.5
    polychromatic_modulator = PolychromaticPhaseModulator(thickness, n=1.5, z=z)
    fig = polychromatic_modulator.visualize(700e-9, show=False, return_fig=True)
    assert isinstance(fig, Figure)
