import pytest
import torch

from torchoptics import Field
from torchoptics.elements import LeftCircularPolarizer, LinearPolarizer, RightCircularPolarizer


def test_linear_polarizer():
    shape = (32, 32)
    theta = torch.tensor(torch.pi / 4)
    spacing = 1
    polarizer = LinearPolarizer(shape, theta, spacing=spacing)
    assert polarizer.shape == shape
    polarized_modulation_profile = polarizer.polarized_modulation_profile()
    expected_matrix = (
        torch.tensor(
            [
                [torch.cos(theta) ** 2, torch.cos(theta) * torch.sin(theta)],
                [torch.cos(theta) * torch.sin(theta), torch.sin(theta) ** 2],
            ],
            dtype=torch.cdouble,
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(2, 2, *shape)
    )
    assert torch.allclose(polarized_modulation_profile[:2, :2], expected_matrix)
    field = Field(torch.ones(4, 3, *shape), wavelength=700e-9, spacing=spacing)
    output_field = polarizer(field)
    assert isinstance(output_field, Field)


def test_left_circular_polarizer():
    shape = (32, 32)
    spacing = 1
    polarizer = LeftCircularPolarizer(shape, spacing=spacing)
    assert polarizer.shape == shape
    polarization_modulation_profile = polarizer.polarized_modulation_profile()
    expected_matrix = (
        torch.tensor([[0.5, -0.5j], [0.5j, 0.5]], dtype=torch.cdouble)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(2, 2, *shape)
    )
    assert torch.allclose(polarization_modulation_profile[:2, :2], expected_matrix)
    field = Field(torch.ones(4, 3, *shape), wavelength=700e-9, spacing=spacing)
    output_field = polarizer(field)
    assert isinstance(output_field, Field)


def test_right_circular_polarizer():
    shape = (32, 32)
    spacing = 1
    polarizer = RightCircularPolarizer(shape, spacing=spacing)
    assert polarizer.shape == shape
    field = Field(torch.ones(4, 3, *shape), wavelength=700e-9, spacing=spacing)
    output_field = polarizer(field)
    assert isinstance(output_field, Field)
