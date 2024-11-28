import unittest

import torch

from torchoptics.config import (
    get_default_spacing,
    get_default_wavelength,
    set_default_spacing,
    set_default_wavelength,
)


class TestConfig(unittest.TestCase):

    def test_set_and_get_default_spacing(self):
        spacing = (10e-6, 10e-6)
        set_default_spacing(spacing)
        result = get_default_spacing()
        self.assertTrue(torch.allclose(result, torch.tensor(spacing, dtype=torch.double)))

    def test_get_default_spacing_not_set(self):
        with self.assertRaises(ValueError):
            get_default_spacing()

    def test_set_and_get_default_wavelength(self):
        wavelength = 700e-6
        set_default_wavelength(wavelength)
        result = get_default_wavelength()
        self.assertTrue(torch.allclose(result, torch.tensor(wavelength, dtype=torch.double)))

    def test_get_default_wavelength_not_set(self):
        with self.assertRaises(ValueError):
            get_default_wavelength()


if __name__ == "__main__":
    unittest.main()
