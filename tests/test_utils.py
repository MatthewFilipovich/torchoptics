import unittest

from torchoptics.utils import *


class TestInitializetensor(unittest.TestCase):
    def test_initialize_tensor(self):
        with self.assertRaises(ValueError):
            initialize_tensor("name", 1.0, is_scalar=True, is_complex=True, is_integer=True)
        with self.assertRaises(ValueError):
            initialize_tensor("name", 1.1, is_scalar=True, is_complex=False, is_integer=True)
