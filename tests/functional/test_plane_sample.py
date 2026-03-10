import torch

import torchoptics
from torchoptics.functional import plane_sample


def test_plane_sample():
    data = torch.arange(12).reshape(3, 4).double()
    data_plane = torchoptics.PlanarGrid((3, 4), 0, 1, None)
    interpolate_plane0 = torchoptics.PlanarGrid((3, 2), 0, 1, None)
    sampled_plane0 = plane_sample(data, data_plane, interpolate_plane0, "bilinear")
    assert torch.allclose(sampled_plane0, torch.tensor([[1, 2], [5, 6], [9, 10]], dtype=torch.double))
    interpolate_plane1 = torchoptics.PlanarGrid((1, 2), 0, 1, None)
    sampled_plane1 = plane_sample(data, data_plane, interpolate_plane1, "bilinear")
    assert torch.allclose(sampled_plane1, torch.tensor([[5, 6]], dtype=torch.double))
    interpolated_plane2 = torchoptics.PlanarGrid((2, 4), 0, (2, 1), None)
    sampled_plane2 = plane_sample(data, data_plane, interpolated_plane2, "bilinear")
    assert torch.allclose(sampled_plane2, torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=torch.double))
    interpolated_plane3 = torchoptics.PlanarGrid((2, 6), 0, (1, 1), None)
    sampled_plane3 = plane_sample(data, data_plane, interpolated_plane3, "bilinear")
    assert torch.allclose(
        sampled_plane3, torch.tensor([[0, 2, 3, 4, 5, 0], [0, 6, 7, 8, 9, 0]], dtype=torch.double)
    )
    interpolated_plane4 = torchoptics.PlanarGrid((2, 4), 0, (0.5, 1), None)
    sampled_plane4 = plane_sample(data, data_plane, interpolated_plane4, "bilinear")
    assert torch.allclose(sampled_plane4, torch.tensor([[3, 4, 5, 6], [5, 6, 7, 8]], dtype=torch.double))
