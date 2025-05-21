import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.special import fresnel

from torchoptics import Field, PlanarGrid
from torchoptics.functional import outer2d
from torchoptics.propagation import VALID_PADDING_MODES, VALID_PROPAGATION_METHODS

# Helper for gaussian_2d


def gaussian_2d(x, y, sigma_x, sigma_y, mu_x, mu_y):
    coefficient = 1 / (2 * torch.pi * sigma_x * sigma_y)
    exponent = -((x - mu_x) ** 2 / (2 * sigma_x**2) + (y - mu_y) ** 2 / (2 * sigma_y**2))
    return coefficient * torch.exp(exponent)


def analytical_square_aperture_field(x, L, N_f, wavelength, propagation_distance):
    S_minus, C_minus = fresnel((2 * N_f) ** 0.5 * (1 - 2 * x / L))
    S_plus, C_plus = fresnel((2 * N_f) ** 0.5 * (1 + 2 * x / L))
    Integral = 1 / 2**0.5 * (C_minus + C_plus) + 1j / 2**0.5 * (S_minus + S_plus)
    xv, yv = np.meshgrid(Integral, Integral)
    field = np.exp(1j * 2 * np.pi / wavelength * propagation_distance) / 1j * xv * yv
    return field


def test_field_initialization():
    shape = (10, 11)
    data = torch.ones(shape, dtype=torch.cdouble)
    z = 5.0
    spacing = 1.0
    offset = None
    wavelength = 0.3
    pg = Field(data, wavelength, z, spacing, offset)
    assert torch.equal(pg.data, torch.ones(shape, dtype=torch.cdouble))
    assert torch.equal(pg.z, torch.tensor(5.0, dtype=torch.double))
    assert torch.equal(pg.spacing, torch.tensor([1.0, 1.0], dtype=torch.double))
    assert torch.equal(pg.offset, torch.tensor([0.0, 0.0], dtype=torch.double))
    assert torch.equal(pg.wavelength, torch.tensor(0.3, dtype=torch.double))
    with pytest.raises(TypeError):
        Field("Wrong type", spacing=1, wavelength=1)
    with pytest.raises(ValueError):
        Field(torch.ones(10), spacing=1, wavelength=1)


def test_field_centroid_and_std():
    shape = (1001, 1000)
    z = 0
    spacing = 0.1753
    offset = (1.63, -0.64)
    wavelength = 1.0
    planar_grid = PlanarGrid(shape, z, spacing, offset)
    x, y = planar_grid.meshgrid()
    sigma_x, sigma_y = 2.6, 1.75
    mu_x, mu_y = -2.34, 3.23
    data = (gaussian_2d(x, y, sigma_x, sigma_y, mu_x, mu_y)) ** 0.5
    field = Field(data.cdouble(), wavelength, z, spacing, offset)
    centroid = field.centroid()
    std = field.std()
    assert torch.allclose(centroid, torch.tensor([mu_x, mu_y], dtype=torch.double), atol=1e-3)
    assert torch.allclose(std, torch.tensor([sigma_x, sigma_y], dtype=torch.double), atol=1e-3)


def test_field_propagation_square_aperture():
    shape = 201
    spacing = 5e-6
    wavelength = 800e-9
    propagation_distance = 0.05
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        for propagation_method in VALID_PROPAGATION_METHODS:
            square_field = torch.ones(shape, shape, device=device)
            input_field = Field(
                square_field.cdouble(),
                spacing=spacing,
                wavelength=wavelength,
            ).to(device)
            output_field = input_field.propagate(
                (shape, shape),
                propagation_distance,
                spacing=spacing,
                propagation_method=propagation_method,
            )
            x = np.linspace(-spacing * shape / 2, spacing * shape / 2, shape)
            L = (shape - 1) * spacing
            N_f = (L / 2) ** 2 / (wavelength * propagation_distance)
            analytical_field = analytical_square_aperture_field(x, L, N_f, wavelength, propagation_distance)
            assert np.allclose(output_field.data.cpu(), analytical_field, atol=1e-1)


def test_field_offset():
    shape = 200
    spacing = 5e-6
    wavelength = 800e-9
    propagation_distance = 0.05
    for propagation_method in VALID_PROPAGATION_METHODS:
        square_field = torch.ones(shape, shape, dtype=torch.cdouble)
        input_field = Field(
            square_field,
            spacing=spacing,
            wavelength=wavelength,
        )
        output_field = input_field.propagate(
            (shape, shape),
            propagation_distance,
            spacing=spacing,
            propagation_method=propagation_method,
        )
        offset = (100 * spacing, -30 * spacing)
        offset_input_field = Field(
            square_field,
            spacing=spacing,
            wavelength=wavelength,
            offset=offset,
        )
        offset_output_field = offset_input_field.propagate(
            (shape, shape), propagation_distance, spacing=spacing, propagation_method=propagation_method
        )
        assert torch.allclose(offset_output_field.data[100:, :-30], output_field.data[:-100, 30:])


def test_field_propagation_methods():
    shape = 201
    spacing = 5e-6
    wavelength = 800e-9
    propagation_distance = 0.05
    square_field = torch.ones(shape, shape, dtype=torch.cdouble)
    input_field = Field(
        square_field,
        spacing=spacing,
        wavelength=wavelength,
    )
    with pytest.raises(TypeError):
        input_field.propagate_to_z(propagation_distance, propagation_method=None)
    with pytest.raises(ValueError):
        input_field.propagate_to_z(propagation_distance, propagation_method="Wrong")


def test_field_asm_propagation():
    shape = 201
    spacing = 5e-6
    wavelength = 800e-9
    propagation_distance = 0.05
    square_field = torch.ones(shape, shape, dtype=torch.cdouble)
    input_field = Field(
        square_field,
        spacing=spacing,
        wavelength=wavelength,
    )
    # Should not fail
    input_field.propagate(
        (shape, shape),
        propagation_distance,
        spacing=input_field.spacing,
        offset=None,
        propagation_method="ASM",
        asm_pad_factor=0,
    )
    with pytest.raises(ValueError):
        input_field.propagate(
            (shape, shape),
            propagation_distance,
            spacing=input_field.spacing,
            offset=(1e-8, 0),
            propagation_method="ASM",
            asm_pad_factor=0,
        )


def test_field_asm_propagation_zero_pad():
    shapes = [(100, 100), (1, 100), (100, 1), (1, 1)]
    spacings = [1e-6, 500e-9]
    wavelength = 700e-9
    propagation_distance = 1
    for shape in shapes:
        for spacing in spacings:
            field = Field(torch.ones(shape, dtype=torch.cdouble), spacing=spacing, wavelength=wavelength)
            field_prop = field.propagate_to_z(
                propagation_distance, propagation_method="asm", asm_pad_factor=0
            )
            assert pytest.approx(field.power().item()) == field_prop.power().item()


def test_field_asm_pad_factor():
    field = Field(torch.ones(10, 10), spacing=1, wavelength=1)
    with pytest.raises(ValueError):
        field.propagate_to_z(1, propagation_method="asm", asm_pad_factor=(1, 2, 3))
    with pytest.raises(ValueError):
        field.propagate_to_z(1, propagation_method="asm", asm_pad_factor=(1, -2))
    with pytest.raises(ValueError):
        field.propagate_to_z(1, propagation_method="asm", asm_pad_factor=(1, 2.2))
    shape = (100, 200)
    spacing = 5e-6
    wavelength = 800e-9
    propagation_distance = 0.05
    asm_pad_factor = (3, 2)
    square_field1 = torch.ones(shape[0], shape[1], dtype=torch.cdouble)
    input_field1 = Field(
        square_field1,
        spacing=spacing,
        wavelength=wavelength,
    )
    square_field2 = torch.zeros(
        (1 + 2 * asm_pad_factor[0]) * shape[0],
        (1 + 2 * asm_pad_factor[1]) * shape[1],
        dtype=torch.cdouble,
    )
    square_field2[
        asm_pad_factor[0] * shape[0] : (asm_pad_factor[0] + 1) * shape[0],
        asm_pad_factor[1] * shape[1] : (asm_pad_factor[1] + 1) * shape[1],
    ] = 1
    input_field2 = Field(
        square_field2,
        spacing=spacing,
        wavelength=wavelength,
    )
    output_field1 = input_field1.propagate(
        (shape[0], shape[1]),
        propagation_distance,
        spacing=spacing,
        propagation_method="ASM",
        asm_pad_factor=asm_pad_factor,
    )
    output_field2 = input_field2.propagate(
        (shape[0], shape[1]),
        propagation_distance,
        spacing=spacing,
        propagation_method="ASM",
        asm_pad_factor=0,
    )
    assert torch.allclose(output_field1.data, output_field2.data)


def test_field_interpolation_modes():
    shape = (100, 100)
    spacing = 1e-6
    wavelength = 500e-9
    data = torch.ones(shape, dtype=torch.cdouble)
    field = Field(data, wavelength, spacing=spacing)
    field_diff_spacing = Field(data, wavelength, spacing=spacing / 3.0)
    for mode in ["nearest", "bilinear", "bicubic"]:
        field.propagate_to_z(1, interpolation_mode=mode)
    with pytest.raises(ValueError):
        field.propagate_to_plane(field_diff_spacing, interpolation_mode="invalid_mode")
    with pytest.raises(TypeError):
        field.propagate_to_plane(field_diff_spacing, interpolation_mode=None)


def test_field_padding_modes():
    shape = (100, 100)
    spacing = 1e-6
    wavelength = 500e-9
    data = torch.ones(shape, dtype=torch.cdouble)
    field = Field(data, wavelength, spacing=spacing)
    field_diff_spacing = Field(data, wavelength, spacing=spacing / 3.0)
    for mode in VALID_PADDING_MODES:
        field.propagate_to_z(1, padding_mode=mode)
    with pytest.raises(ValueError):
        field.propagate_to_plane(field_diff_spacing, padding_mode="invalid_mode")
    with pytest.raises(TypeError):
        field.propagate_to_plane(field_diff_spacing, padding_mode=None)


def test_field_propagate_methods():
    field = Field(torch.ones(10, 10), spacing=1, wavelength=1)
    field_propagate_to_z = field.propagate_to_z(1)
    field_propagate_to_plane = field.propagate_to_plane(PlanarGrid(10, 1, 1))
    field_propagate = field.propagate(10, 1, 1)
    assert torch.allclose(field_propagate_to_z.data, field_propagate.data)
    assert torch.allclose(field_propagate_to_plane.data, field_propagate.data)
    with pytest.raises(TypeError):
        field.propagate_to_plane("Not a PlanarGrid object")


def test_field_modulate():
    field = Field(torch.ones(10, 10), spacing=1, wavelength=1)
    modulated_field = field.modulate(10 * torch.ones(10, 10))
    assert torch.allclose(modulated_field.data, 10 * torch.ones(10, 10, dtype=torch.cdouble))


def test_field_normalization():
    field = Field(torch.rand(10, 10), spacing=10e-6, wavelength=800e-9)
    normalized_field = field.normalize(2)
    assert torch.allclose(normalized_field.power(), torch.tensor(2, dtype=torch.double))


def test_field_inner():
    field = Field(torch.ones(10, 10), spacing=1, wavelength=1)
    inner = field.inner(field)
    assert torch.allclose(inner, torch.tensor(100, dtype=torch.cdouble))
    with pytest.raises(ValueError):
        field.inner(Field(torch.ones(5, 5), spacing=1, wavelength=1))


def test_field_outer():
    field = Field(torch.ones(10, 10), spacing=1, wavelength=1)
    outer = field.outer(field)
    assert torch.allclose(outer, torch.ones(10, 10, 10, 10, dtype=torch.cdouble))
    with pytest.raises(ValueError):
        field.outer(Field(torch.ones(5, 5), spacing=1, wavelength=1))


def test_field_visualize():
    shape = (10, 10)
    data = torch.ones(shape, dtype=torch.cdouble)
    field = Field(data, wavelength=1, spacing=1)
    fig = field.visualize(show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)


def test_field_polarized_split():
    field = Field(torch.ones(3, 10, 10), spacing=1, wavelength=1)
    split_fields = field.polarized_split()
    assert len(split_fields) == 3
    for i, split_field in enumerate(split_fields):
        assert torch.allclose(split_field.data[i], torch.ones(10, 10, dtype=torch.cdouble))
