import pytest
import torch

from torchoptics import Field, PlanarGrid, System
from torchoptics.elements import Detector, IdentityElement, Modulator


def make_system_setup():
    shape = 201
    spacing = 5e-6
    wavelength = 800e-9
    propagation_distance = 0.05
    modulator1_spacing = 5e-6
    modulator1_offset = (0.0, 0.0)
    modulator2_spacing = 6e-6
    modulator2_offset = (1.3e-6, -3.5e-6)
    square_field = torch.ones(shape, shape, dtype=torch.cdouble)
    input_field = Field(square_field, spacing=spacing, wavelength=wavelength)
    mod_profile = torch.ones(shape, shape, dtype=torch.cdouble)
    return (
        shape,
        spacing,
        wavelength,
        propagation_distance,
        modulator1_spacing,
        modulator1_offset,
        modulator2_spacing,
        modulator2_offset,
        square_field,
        input_field,
        mod_profile,
    )


def test_propagation():
    (
        _,
        _,
        _,
        propagation_distance,
        modulator1_spacing,
        modulator1_offset,
        modulator2_spacing,
        modulator2_offset,
        _,
        input_field,
        mod_profile,
    ) = make_system_setup()
    modulator1 = Modulator(mod_profile, propagation_distance, modulator1_spacing, modulator1_offset)
    modulator2 = Modulator(mod_profile, 2 * propagation_distance, modulator2_spacing, modulator2_offset)
    system1 = System(modulator2, modulator1)
    propagated_field = system1(input_field)
    measured_field = system1.measure(input_field, **modulator2.geometry)
    assert propagated_field.is_same_geometry(modulator2)
    assert propagated_field.is_same_geometry(measured_field)
    assert torch.allclose(propagated_field.data, measured_field.data)


def test_elements_along_field_path():
    (
        shape,
        spacing,
        _,
        propagation_distance,
        modulator1_spacing,
        modulator1_offset,
        modulator2_spacing,
        modulator2_offset,
        _,
        input_field,
        mod_profile,
    ) = make_system_setup()
    modulator1 = Modulator(mod_profile, propagation_distance, modulator1_spacing, modulator1_offset)
    modulator2 = Modulator(mod_profile, 2 * propagation_distance, modulator2_spacing, modulator2_offset)
    detector = Detector(shape, z=2 * propagation_distance, spacing=spacing)
    system1 = System(modulator1, detector, modulator2)
    with pytest.raises(TypeError):
        system1(input_field)
    detector.z = 3 * propagation_distance
    system1 = System(modulator1, detector, modulator2)
    system1(input_field)
    system1.measure_at_z(input_field, 2.5 * propagation_distance)
    with pytest.raises(TypeError):
        system1.measure_at_z(input_field, 3 * propagation_distance)
    with pytest.raises(TypeError):
        system1.measure_at_z(input_field, 4 * propagation_distance)
    with pytest.raises(ValueError):
        system1.measure_at_z(input_field, -1)
    system2 = System()
    with pytest.raises(ValueError):
        system2(input_field)
    with pytest.raises(TypeError):
        System(PlanarGrid(shape, z=0, spacing=spacing))


def test_dunder_methods():
    (
        shape,
        _,
        _,
        propagation_distance,
        modulator1_spacing,
        modulator1_offset,
        modulator2_spacing,
        modulator2_offset,
        _,
        _,
        mod_profile,
    ) = make_system_setup()
    modulator1 = Modulator(mod_profile, propagation_distance, modulator1_spacing, modulator1_offset)
    modulator2 = Modulator(mod_profile, 2 * propagation_distance, modulator2_spacing, modulator2_offset)
    detector = Detector(shape, z=2 * propagation_distance, spacing=modulator1_spacing)
    system = System(modulator1, detector, modulator2)
    assert system[0] is modulator1
    assert next(iter(system)) is modulator1
    assert len(system) == 3


def test_measure_at_plane():
    shape, spacing, _, propagation_distance, _, _, _, _, _, input_field, _ = make_system_setup()
    system = System()
    offset = (13e-4, -5e-4)
    plane = PlanarGrid(shape, z=2 * propagation_distance, spacing=spacing, offset=offset)
    measure_plane = system.measure_at_plane(input_field, plane)
    measure = system.measure(input_field, **plane.geometry)
    assert torch.allclose(measure_plane.data, measure.data)


def test_identity_element():
    shape, spacing, _, propagation_distance, _, _, _, _, _, input_field, _ = make_system_setup()
    system = System(IdentityElement(shape, z=propagation_distance, spacing=spacing))
    output_element = Detector(shape, z=2 * propagation_distance, spacing=spacing)
    elements_in_path = system.elements_in_field_path(input_field, output_element)
    assert len(elements_in_path) == 1
    assert elements_in_path[0] is output_element
