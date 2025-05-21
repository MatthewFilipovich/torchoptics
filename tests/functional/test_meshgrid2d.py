import torch

from torchoptics.functional import meshgrid2d


def test_meshgrid2d() -> None:
    bounds = torch.tensor([0.0, 1.0, 0.0, 1.0])
    shape = (2, 2)
    expected_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    expected_y = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    actual_x, actual_y = meshgrid2d(bounds, shape)
    assert torch.allclose(actual_x, expected_x)
    assert torch.allclose(actual_y, expected_y)
