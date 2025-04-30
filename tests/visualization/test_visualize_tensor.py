import matplotlib.pyplot as plt
import pytest
import torch

from torchoptics.visualization.visualization import visualize_tensor


def test_real_tensor():
    tensor = torch.rand(10, 10)
    fig = visualize_tensor(tensor, title="Real Tensor", show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)


def test_complex_tensor():
    tensor = torch.rand(10, 10, dtype=torch.complex64)
    fig = visualize_tensor(tensor, title="Complex Tensor", show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)


def test_invalid_tensor():
    tensor = torch.rand(10, 10, 10)
    with pytest.raises(ValueError):
        visualize_tensor(tensor)


def test_tensor_with_extent():
    tensor = torch.rand(10, 10)
    fig = visualize_tensor(tensor, extent=[0, 1, 0, 1], show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)


def test_tensor_with_vmin_vmax():
    tensor = torch.rand(10, 10)
    fig = visualize_tensor(tensor, vmin=0.2, vmax=0.8, show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)
