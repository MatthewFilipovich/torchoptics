"""This module defines functions for visualizing tensors."""

from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
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
    symbol: Optional[str] = None,
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
        symbol (str, optional): Symbol used in ax title. Default: `None`.
        show (bool, optional): Whether to display the plot. Default: `True`.
        return_fig (bool, optional): Whether to return the figure. Default: `False`.
    """

    if tensor.dim() < 2 or not all(s == 1 for s in tensor.shape[:-2]):  # Check if squeezed tensor is 2D
        raise ValueError(f"Expected tensor to be 2D, but got shape {tensor.shape}.")
    tensor = tensor.detach().cpu().view(tensor.shape[-2], tensor.shape[-1])

    if tensor.is_complex():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        tensor = torch.where(tensor == -0.0 - 0.0j, 0, tensor)  # Remove numerical artifacts

        create_image_subplot(  # Plot absolute square
            tensor.abs().square(),
            extent,
            vmin,
            vmax,
            cmap,
            xlabel,
            ylabel,
            axes[0],
            rf"$|${symbol}$|^2$" if symbol is not None else None,
        )

        create_image_subplot(  # plot angle
            tensor.angle(),
            extent,
            -torch.pi,
            torch.pi,
            "twilight_shifted",
            xlabel,
            ylabel,
            axes[1],
            r"$\arg \{$" + symbol + r"$\}$" if symbol is not None else None,
            cbar_ticks=[-torch.pi, 0, torch.pi],
            cbar_ticklabels=[r"$-\pi$", r"$0$", r"$\pi$"],
        )

        axes[1].get_images()[0].set_interpolation("none")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig, axes = plt.subplots(figsize=(5, 5))
        create_image_subplot(tensor, extent, vmin, vmax, cmap, xlabel, ylabel, axes, symbol)

    if title:
        fig.suptitle(title, y=0.95)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)

    if show:
        plt.show()

    return fig if return_fig else None


def create_image_subplot(  # pylint: disable=too-many-locals
    tensor: Tensor,
    extent: Optional[Sequence[float]],
    vmin: Optional[float],
    vmax: Optional[float],
    cmap: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
    ax: Any,
    ax_title: Optional[str],
    cbar_ticks: Optional[Sequence[float]] = None,
    cbar_ticklabels: Optional[Sequence[str]] = None,
) -> None:
    """Creates a subplot for visualizing a real-valued tensor."""
    extent_tuple = tuple(extent) if extent is not None else None
    im = ax.imshow(tensor, extent=extent_tuple, vmin=vmin, vmax=vmax, cmap=cmap)  # type: ignore[arg-type]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(im, cax=cax, orientation="vertical")
    if cbar_ticks is not None:
        colorbar.set_ticks(cbar_ticks)
    if cbar_ticklabels is not None:
        colorbar.set_ticklabels(cbar_ticklabels)
    ax.set_xlabel(xlabel)  # type: ignore[arg-type]
    ax.set_ylabel(ylabel)  # type: ignore[arg-type]
    ax.set_title(ax_title)
