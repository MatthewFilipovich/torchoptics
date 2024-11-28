import unittest

import matplotlib.pyplot as plt
import torch

from torchoptics.visualization import visualize_tensor


class TestVisualizeTensor(unittest.TestCase):
    def test_real_tensor(self):
        tensor = torch.rand(10, 10)
        fig = visualize_tensor(tensor, title="Real Tensor", show=False, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_complex_tensor(self):
        tensor = torch.rand(10, 10, dtype=torch.complex64)
        fig = visualize_tensor(tensor, title="Complex Tensor", show=False, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_invalid_tensor(self):
        tensor = torch.rand(10, 10, 10)
        with self.assertRaises(ValueError):
            visualize_tensor(tensor)

    def test_tensor_with_extent(self):
        tensor = torch.rand(10, 10)
        fig = visualize_tensor(tensor, extent=[0, 1, 0, 1], show=False, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_tensor_with_vmin_vmax(self):
        tensor = torch.rand(10, 10)
        fig = visualize_tensor(tensor, vmin=0.2, vmax=0.8, show=False, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)


if __name__ == "__main__":
    unittest.main()
