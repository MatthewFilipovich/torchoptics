import unittest

import torch
from torch import Tensor
from torch.nn import Parameter

from torchoptics import OpticsModule


class TestOpticsModule(unittest.TestCase):

    def test_initialization(self):
        module = OpticsModule()
        self.assertIsInstance(module, OpticsModule)

    def test_register_property_as_parameter(self):
        module = OpticsModule()
        value = torch.tensor([1.0, 2.0, 3.0])
        module.register_optics_property("prop1", Parameter(value))
        self.assertTrue(hasattr(module, "prop1"))
        self.assertIsInstance(module.prop1, Tensor)
        self.assertTrue(module.prop1.requires_grad)
        self.assertIn("prop1", dict(module.named_parameters()))
        self.assertTrue(torch.equal(module.prop1, value))
        self.assertFalse(module.prop1 is value)  # Ensure a copy is made

    def test_register_property_as_buffer(self):
        module = OpticsModule()
        value = torch.tensor([1.0, 2.0, 3.0])
        module.register_optics_property("prop1", value)
        self.assertTrue(hasattr(module, "prop1"))
        self.assertIsInstance(module.prop1, Tensor)
        self.assertFalse(module.prop1.requires_grad)
        self.assertIn("prop1", dict(module.named_buffers()))

    def test_set_property(self):
        module = OpticsModule()
        initial_value = torch.tensor([1.0, 2.0, 3.0])
        module.register_optics_property("prop1", initial_value)
        new_value = torch.tensor([4.0, 5.0, 6.0])
        module.set_optics_property("prop1", new_value)
        self.assertTrue(torch.equal(module.prop1, new_value))

    def test_property_shape_validation(self):
        module = OpticsModule()
        initial_value = torch.tensor([1.0, 2.0, 3.0])
        module.register_optics_property("prop1", initial_value)
        new_value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            module.set_optics_property("prop1", new_value)

    def test_complex_property_registration(self):
        module = OpticsModule()
        value = torch.tensor([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
        module.register_optics_property("prop1", value, is_complex=True)
        self.assertTrue(module.prop1.is_complex())

    def test_is_positive(self):
        module = OpticsModule()
        value = torch.tensor([1.0, 2.0, 3.0])
        module.register_optics_property("prop1", value, is_positive=True)
        invalid_value = torch.tensor([-1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            module.set_optics_property("prop1", invalid_value)

    def test_register_property_before_init(self):
        module = OpticsModule()
        module._initialized = False
        with self.assertRaises(AttributeError):
            module.register_optics_property("prop1", torch.tensor([1.0]))

    def test_register_property_from_sequence(self):
        module = OpticsModule()
        value = [1.0, 2.0]
        module.register_optics_property("prop1", value, is_vector2=True)
        self.assertTrue(hasattr(module, "prop1"))
        self.assertIsInstance(module.prop1, Tensor)
        expected_tensor = torch.tensor([1.0, 2.0], dtype=torch.double)
        self.assertTrue(torch.equal(module.prop1, expected_tensor))
        self.assertFalse(module.prop1.requires_grad)
        self.assertIn("prop1", dict(module.named_buffers()))

    def test_register_property_with_none_shape(self):
        module = OpticsModule()
        value = torch.tensor([1.0, 2.0, 3.0])
        module.register_optics_property("prop1", value)
        self.assertTrue(hasattr(module, "prop1"))
        self.assertIsInstance(module.prop1, Tensor)
        self.assertFalse(module.prop1.requires_grad)
        self.assertEqual(module.prop1.shape, (3,))

    def test_register_trainable_property_with_none_shape(self):
        module = OpticsModule()
        value = torch.tensor([1.0, 2.0, 3.0])
        module.register_optics_property("prop1", Parameter(value))
        self.assertTrue(hasattr(module, "prop1"))
        self.assertIsInstance(module.prop1, Tensor)
        self.assertTrue(module.prop1.requires_grad)
        self.assertEqual(module.prop1.shape, (3,))

    def test_set_property_via_setattr(self):
        module = OpticsModule()
        initial_value = [1.0, 2.0]
        module.register_optics_property("prop1", initial_value, is_vector2=True)
        new_value = [4.0, 5.0]
        module.prop1 = new_value  # Using setattr
        expected_tensor = torch.tensor(new_value, dtype=torch.double)
        self.assertTrue(torch.equal(module.prop1, expected_tensor))

    def test_set_property_via_setattr_tensor(self):
        module = OpticsModule()
        initial_value = [1.0, 2.0, 3.0]
        module.register_optics_property("prop1", initial_value)
        new_value = torch.tensor([4.0, 5.0, 6.0])
        module.prop1 = new_value  # Using setattr
        self.assertTrue(torch.equal(module.prop1, new_value))

    def test_set_trainable_property_via_setattr(self):
        module = OpticsModule()
        initial_value = torch.tensor([1.0, 2.0, 3.0])
        module.register_optics_property("prop1", Parameter(initial_value))
        new_value = [4.0, 5.0, 6.0]
        # Prevents RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
        with torch.no_grad():
            module.prop1 = new_value  # Using setattr
        expected_tensor = torch.tensor(new_value, dtype=torch.double)
        self.assertTrue(torch.equal(module.prop1, expected_tensor))

    def test_raise_errors(self):
        with self.assertRaises(AttributeError):
            OpticsModule().set_optics_property("unregistered_prop", 1.0)


if __name__ == "__main__":
    unittest.main()
