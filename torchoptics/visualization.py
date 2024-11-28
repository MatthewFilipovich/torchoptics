"""This module defines functions for visualizing tensors."""

from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

__all__ = ["visualize_tensor"]


def visualize_tensor(
    tensor: Tensor,
    title: Optional[str] = None,
    extent: Optional[Sequence[float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "inferno",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Visualizes a 2D real or complex-valued tensor.

    Args:
        tensor (Tensor): The 2D tensor to visualize.
        title (str, optional): The title of the plot. Default: `None`.
        extent (Sequence[float], optional): The bounding box in data coordinates that the image will fill
            (left, right, bottom, top). Default: `None`.
        vmin (float, optional): The minimum value of the color scale. Default: `None`.
        vmax (float, optional): The maximum value of the color scale. Default: `None`.
        cmap (str, optional): The colormap to use. Default: `"inferno"`.
        xlabel (str, optional): The label for the x-axis. Default: `None`.
        ylabel (str, optional): The label for the y-axis. Default: `None`.
        show (bool, optional): Whether to display the plot. Default: `True`.
        return_fig (bool, optional): Whether to return the figure. Default: `False`.
    """

    if not all(s == 1 for s in tensor.shape[:-2]):  # Check if squeezed tensor is 2D
        raise ValueError(f"Expected tensor to be 2D, but got shape {tensor.shape}.")
    tensor = tensor.detach().cpu().view(tensor.shape[-2], tensor.shape[-1])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10)) if tensor.is_complex() else plt.subplots(figsize=(5, 5))

    subplots_func = _create_complex_image_subplots if tensor.is_complex() else _create_image_subplot
    subplots_func(tensor, extent, vmin, vmax, cmap, xlabel, ylabel, axes)

    if title:
        fig.suptitle(title, y=0.95)

    if show:
        plt.show()

    return fig if return_fig else None


def _create_complex_image_subplots(
    tensor: Tensor,
    extent: Optional[Sequence[float]],
    vmin: Optional[float],
    vmax: Optional[float],
    cmap: str,
    xlabel: Optional[str],
    ylabel: Optional[str],
    axes: Any,
) -> None:
    """Creates subplots for visualizing a complex-valued tensor."""
    components = [tensor.abs().square(), tensor.angle(), tensor.real, tensor.imag]
    ax_titles = [r"$|\psi|^2$", r"$\arg \{ \psi \}$", r"$\Re \{\psi \}$", r"$\Im \{\psi \}$"]
    cmap_list = [cmap, "twilight_shifted", "viridis", "viridis"]
    vmin_list = [vmin, -torch.pi, None, None]
    vmax_list = [vmax, torch.pi, None, None]

    for i in range(4):
        _create_image_subplot(
            components[i],
            extent=extent,
            vmin=vmin_list[i],
            vmax=vmax_list[i],
            cmap=cmap_list[i],
            xlabel=xlabel,
            ylabel=ylabel,
            ax=axes.flat[i],  # type: ignore[attr-defined]
        )
        axes.flat[i].set_title(ax_titles[i])  # type: ignore[attr-defined]

    axes[0, 1].get_images()[0].set_interpolation("none")  # type: ignore[index]
    plt.subplots_adjust(wspace=0.4, hspace=0.4)


def _create_image_subplot(
    tensor: Tensor,
    extent: Optional[Sequence[float]],
    vmin: Optional[float],
    vmax: Optional[float],
    cmap: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
    ax: Any,
) -> None:
    """Creates a subplot for visualizing a real-valued tensor."""
    extent_tuple = tuple(extent) if extent is not None else None
    im = ax.imshow(tensor, extent=extent_tuple, vmin=vmin, vmax=vmax, cmap=cmap)  # type: ignore[arg-type]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation="vertical")
    ax.set_xlabel(xlabel)  # type: ignore[arg-type]
    ax.set_ylabel(ylabel)  # type: ignore[arg-type]
