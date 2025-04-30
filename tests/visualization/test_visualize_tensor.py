import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.animation import Animation, FuncAnimation

from torchoptics.visualization.visualization import animate_tensor, visualize_tensor


def test_real_tensor():
    tensor = torch.rand(10, 10)
    fig = visualize_tensor(tensor, title="Real Tensor", show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Main plot and colorbar


def test_complex_tensor():
    tensor = torch.rand(10, 10, dtype=torch.complex64)
    fig = visualize_tensor(tensor, title="Complex Tensor", show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4  # two subplots for magnitude and phase, and two colorbar


def test_invalid_tensor_shape_3d():
    tensor = torch.rand(10, 10, 10)
    with pytest.raises(ValueError):
        visualize_tensor(tensor)


def test_visualize_tensor_all_options():
    tensor = torch.rand(10, 10)

    fig = visualize_tensor(
        tensor,
        extent=[0, 1, 0, 1],
        xlabel="X",
        ylabel="Y",
        title="Labeled Tensor",
        vmin=0.2,
        vmax=0.8,
        symbol="u",
        show=False,
        return_fig=True,
    )

    assert isinstance(fig, plt.Figure)


def test_animate_real_tensor():
    tensor = torch.rand(5, 10, 10)
    anim = animate_tensor(tensor, title="Real Animation", show=False)
    assert isinstance(anim, FuncAnimation)


def test_animate_complex_tensor():
    tensor = torch.rand(5, 10, 10, dtype=torch.complex64)
    anim = animate_tensor(tensor, title="Complex Animation", show=False)
    assert isinstance(anim, FuncAnimation)


def test_animate_tensor_invalid_shape():
    tensor = torch.rand(10, 10, 10, 10)
    with pytest.raises(ValueError):
        animate_tensor(tensor)


def test_animate_tensor_with_titles_vmins_vmaxes():
    tensor = torch.rand(5, 10, 10)  # 5 frames of 10x10 tensors

    titles = [f"Frame {i}" for i in range(5)]
    vmins = torch.arange(5)
    vmaxs = torch.arange(5) + 1

    anim = animate_tensor(
        tensor,
        title=titles,
        vmin=vmins,
        vmax=vmaxs,
        show=False,
    )

    assert isinstance(anim, Animation)


def test_animate_tensor_with_custom_kwargs():
    tensor = torch.rand(5, 10, 10)
    anim = animate_tensor(tensor, func_anim_kwargs={"interval": 500}, show=False)
    assert isinstance(anim, FuncAnimation)
