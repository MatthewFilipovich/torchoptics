import unittest

import torch

from torchoptics.utils import *


class TestInitializetensor(unittest.TestCase):
    def test_initialize_tensor(self):
        with self.assertRaises(ValueError):
            initialize_tensor("name", 1.0, is_scalar=True, is_complex=True, is_integer=True)
        with self.assertRaises(ValueError):
            initialize_tensor("name", 1.1, is_scalar=True, is_complex=False, is_integer=True)

    def test_initialize_scalar_tensor(self):
        tensor = initialize_tensor("scalar", 1.0, is_scalar=True)
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tensor.item(), 1.0)

    def test_scalar_and_vector2(self):
        with self.assertRaises(ValueError):
            initialize_tensor("scalar_and_vector2", 1.0, is_scalar=True, is_vector2=True)

    def test_initialize_vector2_tensor(self):
        tensor = initialize_tensor("vector2", [1.0, 2.0], is_vector2=True)
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

    def test_initialize_complex_tensor(self):
        tensor = initialize_tensor("complex", 1.0 + 2.0j, is_complex=True)
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tensor.item(), 1.0 + 2.0j)

    def test_initialize_integer_tensor(self):
        tensor = initialize_tensor("integer", 1, is_integer=True)
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tensor.item(), 1)

    def test_initialize_positive_tensor(self):
        tensor = initialize_tensor("positive", 1.0, is_positive=True)
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tensor.item(), 1.0)

    def test_initialize_non_negative_tensor(self):
        tensor = initialize_tensor("non_negative", 0.0, is_non_negative=True)
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tensor.item(), 0.0)

    def test_initialize_tensor_invalid_scalar(self):
        with self.assertRaises(ValueError):
            initialize_tensor("invalid_scalar", [1.0, 2.0], is_scalar=True)

    def test_initialize_tensor_invalid_vector2(self):
        with self.assertRaises(ValueError):
            initialize_tensor("invalid_vector2", [1.0, 2.0, 3.0], is_vector2=True)

    def test_initialize_tensor_invalid_positive(self):
        with self.assertRaises(ValueError):
            initialize_tensor("invalid_positive", -1.0, is_positive=True)

    def test_initialize_tensor_invalid_non_negative(self):
        with self.assertRaises(ValueError):
            initialize_tensor("invalid_non_negative", -1.0, is_non_negative=True)


if __name__ == "__main__":
    unittest.main()
