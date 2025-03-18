import unittest
from unittest.mock import patch

import matplotlib
import torch

import torchoptics
from torchoptics import PlanarGeometry


class TestPlanarGeometry(unittest.TestCase):

    def setUp(self):
        # Default parameters for testing
        self.shape = (100, 101)
        self.z = 1.0
        self.spacing = (0.1, 0.2)
        self.offset = (0.0, 0.0)

    def test_initialization(self):
        plane = PlanarGeometry(
            shape=self.shape,
            z=self.z,
            spacing=self.spacing,
            offset=self.offset,
        )

        self.assertEqual(plane.shape, self.shape)
        self.assertTrue(torch.equal(plane.z, torch.tensor(self.z, dtype=torch.double)))
        self.assertTrue(torch.equal(plane.spacing, torch.tensor(self.spacing, dtype=torch.double)))
        self.assertTrue(torch.equal(plane.offset, torch.tensor(self.offset, dtype=torch.double)))

    def test_shape(self):
        plane = PlanarGeometry(
            shape=self.shape,
            z=self.z,
            spacing=self.spacing,
            offset=self.offset,
        )

        self.assertIsInstance(plane.shape, tuple)
        self.assertEqual(len(plane.shape), 2)
        self.assertTrue(all(isinstance(s, int) for s in plane.shape))
        print(plane.shape)

    def test_default_initialization(self):
        torchoptics.set_default_spacing((0.1, 0.1))
        plane = PlanarGeometry(shape=self.shape, z=self.z)
        self.assertTrue(torch.equal(plane.spacing, torch.tensor((0.1, 0.1), dtype=torch.double)))

    def test_geometry_property(self):
        plane = PlanarGeometry(shape=self.shape, z=self.z, spacing=self.spacing, offset=self.offset)

        expected_geometry = {
            "shape": self.shape,
            "z": torch.tensor(self.z, dtype=torch.double),
            "spacing": torch.tensor(self.spacing, dtype=torch.double),
            "offset": torch.tensor(self.offset, dtype=torch.double),
        }

        self.assertEqual(plane.geometry["shape"], expected_geometry["shape"])
        self.assertTrue(torch.equal(plane.geometry["z"], expected_geometry["z"]))
        self.assertTrue(torch.equal(plane.geometry["spacing"], expected_geometry["spacing"]))
        self.assertTrue(torch.equal(plane.geometry["offset"], expected_geometry["offset"]))

    def test_grid_cell_area(self):
        plane = PlanarGeometry(shape=self.shape, z=self.z, spacing=self.spacing, offset=self.offset)

        expected_area = torch.tensor(self.spacing[0] * self.spacing[1], dtype=torch.double)
        self.assertTrue(torch.equal(plane.cell_area(), expected_area))

    def test_extent(self):
        plane = PlanarGeometry(shape=self.shape, z=self.z, spacing=self.spacing, offset=self.offset)

        expected_extent = torch.tensor(self.spacing, dtype=torch.double) * (
            torch.tensor(self.shape, dtype=torch.double) - 1
        )
        self.assertTrue(torch.equal(plane.length(True), expected_extent))

    def test_bounds(self):
        plane = PlanarGeometry(shape=self.shape, z=self.z, spacing=self.spacing, offset=self.offset)

        half_length = plane.length() / 2
        expected_bounds = torch.tensor(
            [
                self.offset[0] - half_length[0],
                self.offset[0] + half_length[0],
                self.offset[1] - half_length[1],
                self.offset[1] + half_length[1],
            ]
        )
        self.assertTrue(torch.equal(plane.bounds(), expected_bounds))

    def test_meshgrid(self):
        plane = PlanarGeometry(shape=self.shape, z=self.z, spacing=self.spacing, offset=self.offset)

        x, y = plane.meshgrid()
        self.assertEqual(x.shape, self.shape)
        self.assertEqual(y.shape, self.shape)

    def test_is_same_geometry(self):
        pg1 = PlanarGeometry((10, 10), 5.0, (1.0, 1.0), (0.0, 0.0))
        pg2 = PlanarGeometry((10, 10), 5.0, (1.0, 1.0), (0.0, 0.0))
        pg3 = PlanarGeometry((10, 10), 5.0, (2.0, 2.0), (1.0, 1.0))

        self.assertTrue(pg1.is_same_geometry(pg2))
        self.assertFalse(pg1.is_same_geometry(pg3))

    @patch("matplotlib.pyplot.show")
    def test_visualize(self, mock_show):
        plane = PlanarGeometry(shape=self.shape, z=self.z, spacing=self.spacing, offset=self.offset)

        tensor = torch.randn(self.shape)

        visual = plane._visualize(tensor, show=True, return_fig=True, show_bounds=True)

        mock_show.assert_called_once()  # Check if plt.show() was called
        self.assertIsInstance(visual, matplotlib.pyplot.Figure)

    def test_repr(self):
        plane = PlanarGeometry(shape=self.shape, z=self.z, spacing=self.spacing, offset=self.offset)

        expected_repr = "PlanarGeometry(shape=(100, 101), z=1.00e+00, spacing=(1.00e-01, 2.00e-01), offset=(0.00e+00, 0.00e+00))"
        self.assertEqual(repr(plane), expected_repr)


if __name__ == "__main__":
    unittest.main()
