import unittest

import torch

import torchoptics
from torchoptics.functional import *


class TestConv2d(unittest.TestCase):
    def test_conv2d_fft(self):
        input = torch.randn(3, 1, 11, 23) + 1j * torch.randn(3, 1, 11, 23)
        weight = torch.randn(1, 1, 3, 5) + 1j * torch.randn(1, 1, 3, 5)
        # Need to flip the weight for torch conv2d, otherwise it's the cross-correlation function
        conv2d_output = torch.nn.functional.conv2d(input, weight.flip(-1, -2))
        conv2d_fft_output = conv2d_fft(input, weight)

        self.assertTrue(torch.allclose(conv2d_output, conv2d_fft_output, atol=1e-5))


class TestPlaneSample(unittest.TestCase):
    def test_plane_sample(self):
        data = torch.arange(12).reshape(3, 4).double()
        data_plane = torchoptics.PlanarGeometry((3, 4), 0, 1, None)

        interpolate_plane0 = torchoptics.PlanarGeometry((3, 2), 0, 1, None)
        sampled_plane0 = plane_sample(data, data_plane, interpolate_plane0, "bilinear")
        self.assertTrue(
            torch.allclose(sampled_plane0, torch.tensor([[1, 2], [5, 6], [9, 10]], dtype=torch.double))
        )

        interpolate_plane1 = torchoptics.PlanarGeometry((1, 2), 0, 1, None)
        sampled_plane1 = plane_sample(data, data_plane, interpolate_plane1, "bilinear")
        self.assertTrue(torch.allclose(sampled_plane1, torch.tensor([[5, 6]], dtype=torch.double)))

        interpolated_plane2 = torchoptics.PlanarGeometry((2, 4), 0, (2, 1), None)
        sampled_plane2 = plane_sample(data, data_plane, interpolated_plane2, "bilinear")
        self.assertTrue(
            torch.allclose(sampled_plane2, torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=torch.double))
        )

        interpolated_plane3 = torchoptics.PlanarGeometry((2, 6), 0, (1, 1), None)
        sampled_plane3 = plane_sample(data, data_plane, interpolated_plane3, "bilinear")
        self.assertTrue(
            torch.allclose(
                sampled_plane3, torch.tensor([[0, 2, 3, 4, 5, 0], [0, 6, 7, 8, 9, 0]], dtype=torch.double)
            )
        )

        interpolated_plane4 = torchoptics.PlanarGeometry((2, 4), 0, (0.5, 1), None)
        sampled_plane4 = plane_sample(data, data_plane, interpolated_plane4, "bilinear")
        self.assertTrue(
            torch.allclose(sampled_plane4, torch.tensor([[3, 4, 5, 6], [5, 6, 7, 8]], dtype=torch.double))
        )


class TestFFTFreqGrad(unittest.TestCase):

    def test_fftfreq_grad_values(self):
        """Test that fftfreq_grad matches torch.fft.fftfreq for various inputs."""
        for n in [0, 1, 2, 3, 8, 16, 31, 32]:  # Different sizes
            for d_value in [1e-9, 1e-3, 1e-1, 1e1, 1e7]:  # Different spacings
                with self.subTest(n=n, d=d_value):
                    d = torch.tensor(d_value)
                    expected = torch.fft.fftfreq(n, d=d_value)
                    actual = fftfreq_grad(n, d)
                    self.assertTrue(torch.allclose(actual, expected))

    def test_fftfreq_grad_dtype_device(self):
        """Test that fftfreq_grad supports different data types and devices."""
        n = 16
        d_value = 0.17
        dtype = torch.double
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for requires_grad in [False, True]:
                with self.subTest(dtype=dtype, device=device):
                    d = torch.tensor(d_value, dtype=dtype, device=device, requires_grad=requires_grad)
                    expected = torch.fft.fftfreq(n, d=d_value).to(dtype=dtype, device=device)
                    actual = fftfreq_grad(n, d)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.device, expected.device)
                    self.assertEqual(actual.requires_grad, requires_grad)
                    self.assertTrue(torch.allclose(actual, expected))


class TestLinspaceGrad(unittest.TestCase):

    def test_output_values(self):
        """Test that linspace_grad produces correct linearly spaced values."""
        start_val, end_val = 0.0, 1.0
        steps = 10
        start = torch.tensor(start_val, requires_grad=True, dtype=torch.float64)
        end = torch.tensor(end_val, requires_grad=True, dtype=torch.float64)
        expected = torch.linspace(start_val, end_val, steps, dtype=torch.float64)
        actual = linspace_grad(start, end, steps)
        self.assertTrue(torch.allclose(actual, expected))

    def test_edge_cases(self):
        for start_val, end_val, steps in [(1, 1, 0), (1, 3, 0), (12, 12, 1), (12, 14, 1)]:
            with self.subTest(start_val=start_val, end_val=end_val, steps=steps):
                start = torch.tensor(start_val, requires_grad=True, dtype=torch.float64)
                end = torch.tensor(end_val, requires_grad=True, dtype=torch.float64)
                expected = torch.linspace(start_val, end_val, steps, dtype=torch.float64)
                actual = linspace_grad(start, end, steps)
                self.assertTrue(torch.allclose(actual, expected))

    def test_differentiability(self):
        """Test that gradients can flow back through start and end tensors."""
        start = torch.tensor(-3, requires_grad=True, dtype=torch.float64)
        end = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
        steps = 10
        linspace_tensor = linspace_grad(start, end, steps)
        output = linspace_tensor.sum()
        output.backward()
        self.assertIsNotNone(start.grad)
        self.assertIsNotNone(end.grad)

    def test_dtype(self):
        """Test handling of different devices and data types."""
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for dtype in [torch.float32, torch.float64]:
                with self.subTest(device=device, dtype=dtype):
                    start = torch.tensor(0.0, requires_grad=True, device=device, dtype=dtype)
                    end = torch.tensor(1.0, requires_grad=True, device=device, dtype=dtype)
                    steps = 10
                    linspace_tensor = linspace_grad(start, end, steps)
                    self.assertEqual(linspace_tensor.device, start.device)
                    self.assertEqual(linspace_tensor.dtype, dtype)


class TestMeshgrid2D(unittest.TestCase):

    def test_meshgrid2d(self):
        bounds = torch.tensor([0.0, 1.0, 0.0, 1.0])
        shape = (2, 2)
        expected_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        expected_y = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        actual_x, actual_y = meshgrid2d(bounds, shape)
        self.assertTrue(torch.allclose(actual_x, expected_x))
        self.assertTrue(torch.allclose(actual_y, expected_y))


if __name__ == "__main__":
    unittest.main()
