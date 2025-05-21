import torch

from torchoptics import Field
from torchoptics.elements import HalfWaveplate, QuarterWaveplate, Waveplate


def test_waveplate_forward() -> None:
    shape = (32, 32)
    phi = torch.tensor(torch.pi / 2)
    theta = torch.tensor(torch.pi / 4)
    spacing = 1
    waveplate = Waveplate(shape, phi, theta, spacing=spacing)
    field = Field(torch.ones(4, 3, *shape), wavelength=700e-9, spacing=spacing)
    output_field = waveplate(field)
    assert isinstance(output_field, Field)


def test_waveplate_modulation_profile() -> None:
    shape = (32, 32)
    phi = torch.tensor(0.0)
    theta = torch.tensor(0.0)
    spacing = 1
    waveplate = Waveplate(shape, phi, theta, spacing=spacing)
    polarized_modulation_profile = waveplate.polarized_modulation_profile()
    expected_matrix = (
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.cdouble,
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(2, 2, *shape)
    )
    assert torch.allclose(polarized_modulation_profile[:2, :2], expected_matrix)


def test_waveplate_profile() -> None:
    shape = (32, 30)
    phi = torch.tensor(torch.pi / 2)
    theta = torch.tensor(torch.pi / 4)
    spacing = 1
    waveplate = Waveplate(shape, phi, theta, spacing=spacing)
    polarized_modulation_profile = waveplate.polarized_modulation_profile()
    expected_matrix = 0.5 * torch.tensor(
        [
            [1 + 1j, 1 - 1j, 0],
            [1 - 1j, 1 + 1j, 0],
            [0, 0, 2],
        ],
        dtype=torch.cdouble,
    ).unsqueeze(-1).unsqueeze(-1).expand(3, 3, *shape)
    assert torch.allclose(polarized_modulation_profile, expected_matrix)


def test_quarter_waveplate_profile() -> None:
    shape = (32, 32)
    theta = torch.tensor(torch.pi / 4)
    spacing = 1
    qwp = QuarterWaveplate(shape, theta, spacing=spacing)
    waveplate = Waveplate(shape, torch.pi / 2, theta, spacing=spacing)
    assert torch.allclose(qwp.polarized_modulation_profile(), waveplate.polarized_modulation_profile())


def test_half_waveplate_profile() -> None:
    shape = (32, 32)
    theta = torch.tensor(torch.pi / 4)
    spacing = 1
    hwp = HalfWaveplate(shape, theta, spacing=spacing)
    waveplate = Waveplate(shape, torch.pi, theta, spacing=spacing)
    assert torch.allclose(hwp.polarized_modulation_profile(), waveplate.polarized_modulation_profile())
