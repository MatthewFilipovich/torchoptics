import unittest

import torch

from torchoptics import Field, PlanarGeometry, System
from torchoptics.elements import Detector, IdentityElement, Modulator


class TestSystem(unittest.TestCase):

    def setUp(self):
        """Common setup for all tests"""
        self.shape = 201
        self.spacing = 5e-6
        self.wavelength = 800e-9
        self.propagation_distance = 0.05
        self.modulator1_spacing = 5e-6
        self.modulator1_offset = (0.0, 0.0)  # Assuming no offset for modulator1
        self.modulator2_spacing = 6e-6
        self.modulator2_offset = (1.3e-6, -3.5e-6)

        # Common field setup
        self.square_field = torch.ones(self.shape, self.shape, dtype=torch.cdouble)
        self.input_field = Field(
            self.square_field,
            spacing=self.spacing,
            wavelength=self.wavelength,
        )

        self.mod_profile = torch.ones(self.shape, self.shape, dtype=torch.cdouble)

    def test_propagation(self):
        """Test propagation through a system"""
        modulator1 = Modulator(
            self.mod_profile, self.propagation_distance, self.modulator1_spacing, self.modulator1_offset
        )
        modulator2 = Modulator(
            self.mod_profile, 2 * self.propagation_distance, self.modulator2_spacing, self.modulator2_offset
        )
        system1 = System(modulator2, modulator1)

        propagated_field = system1(self.input_field)
        measured_field = system1.measure(self.input_field, **modulator2.geometry)

        self.assertTrue(propagated_field.is_same_geometry(modulator2), "Geometry mismatch after propagation")
        self.assertTrue(
            propagated_field.is_same_geometry(measured_field),
            "Geometry mismatch between propagated and measured field",
        )
        self.assertTrue(
            torch.allclose(propagated_field.data, measured_field.data),
            "Field data mismatch between propagated and measured field",
        )

    def test_elements_along_field_path(self):
        """Test the sequence and placement of elements along the field path"""
        # Create modulators and detector
        modulator1 = Modulator(
            self.mod_profile, self.propagation_distance, self.modulator1_spacing, self.modulator1_offset
        )
        modulator2 = Modulator(
            self.mod_profile, 2 * self.propagation_distance, self.modulator2_spacing, self.modulator2_offset
        )
        detector = Detector(self.shape, z=2 * self.propagation_distance, spacing=self.spacing)

        # Create system with modulators and detector
        system1 = System(modulator1, detector, modulator2)

        with self.assertRaises(TypeError):  # Raises error because detector is not at end of system
            system1(self.input_field)

        # Should not raise an error when detector is placed after the propagation distance
        detector.z = 3 * self.propagation_distance
        system1 = System(modulator1, detector, modulator2)
        system1(self.input_field)
        system1.measure_at_z(self.input_field, 2.5 * self.propagation_distance)

        with self.assertRaises(TypeError):
            system1.measure_at_z(self.input_field, 3 * self.propagation_distance)
        with self.assertRaises(TypeError):
            system1.measure_at_z(self.input_field, 4 * self.propagation_distance)
        with self.assertRaises(ValueError):
            system1.measure_at_z(self.input_field, -1)

        system2 = System()
        with self.assertRaises(ValueError):
            system2(self.input_field)  # Empty system should raise an error

        system3 = System(PlanarGeometry(self.shape, z=0, spacing=self.spacing))
        with self.assertRaises(TypeError):
            system3(self.input_field)  # PlanarGeometry should not be used as an element

    def test_dunder_methods(self):
        modulator1 = Modulator(
            self.mod_profile, self.propagation_distance, self.modulator1_spacing, self.modulator1_offset
        )
        modulator2 = Modulator(
            self.mod_profile, 2 * self.propagation_distance, self.modulator2_spacing, self.modulator2_offset
        )
        detector = Detector(self.shape, z=2 * self.propagation_distance, spacing=self.spacing)
        system = System(modulator1, detector, modulator2)

        self.assertTrue(system[0] is modulator1)
        self.assertTrue(next(iter(system)) is modulator1)
        self.assertTrue(len(system) == 3)

    def test_measure_at_plane(self):
        system = System()
        offset = (13e-4, -5e-4)
        plane = PlanarGeometry(
            self.shape, z=2 * self.propagation_distance, spacing=self.spacing, offset=offset
        )
        measure_plane = system.measure_at_plane(self.input_field, plane)
        measure = system.measure(self.input_field, **plane.geometry)
        self.assertTrue(torch.allclose(measure_plane.data, measure.data))

    def test_identity_element(self):
        """Test identity element is removed from the path at output"""
        system = System(IdentityElement(self.shape, z=self.propagation_distance, spacing=self.spacing))
        output_element = Detector(self.shape, z=2 * self.propagation_distance, spacing=self.spacing)
        elements_in_path = system.elements_in_field_path(self.input_field, output_element)

        self.assertTrue(len(elements_in_path) == 1)
        self.assertTrue(elements_in_path[0] is output_element)


if __name__ == "__main__":
    unittest.main()
