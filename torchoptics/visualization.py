"""Visualization utilities for real or complex-valued tensors using matplotlib."""

from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from torch import Tensor

__all__ = ["visualize_tensor", "animate_tensor"]
# pylint: disable=too-many-locals


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
    interpolation: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Visualize a 2D real or complex-valued tensor using matplotlib.

    If the tensor is complex, two subplots are shown: one for the magnitude squared and one for the phase.

    Args:
        tensor (Tensor): A 2D tensor of shape (H, W).
        title (str, optional): Title for the figure.
        extent (Sequence[float], optional): Bounding box in data coordinates (left, right, bottom, top).
        vmin (float, optional): Minimum value for color scaling.
        vmax (float, optional): Maximum value for color scaling.
        cmap (str, optional): Colormap for the magnitude or real plot. Defaults to "inferno".
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        symbol (str, optional): Symbol used in subplot titles for LaTeX rendering.
        interpolation (str, optional): Interpolation method for imshow.
        show (bool, optional): Whether to call `plt.show()`. Defaults to True.
        return_fig (bool, optional): If True, returns the matplotlib Figure.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure if `return_fig` is True, else None.
    """
    if tensor.ndim < 2 or not all(s == 1 for s in tensor.shape[:-2]):
        raise ValueError(f"Expected tensor to be 2D, but got shape {tensor.shape}.")

    tensor = tensor.detach().cpu().view(tensor.shape[-2], tensor.shape[-1])

    if tensor.is_complex():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        tensor = torch.where(tensor == -0.0 - 0.0j, 0, tensor)

        create_image_subplot(  # Plot magnitude squared
            axes[0],
            tensor.abs().square(),
            extent,
            vmin,
            vmax,
            cmap,
            xlabel,
            ylabel,
            rf"$|${symbol}$|^2$" if symbol else None,
            interpolation,
        )

        create_image_subplot(  # Plot phase
            axes[1],
            tensor.angle(),
            extent,
            -torch.pi,
            torch.pi,
            "twilight_shifted",
            xlabel,
            ylabel,
            r"$\arg \{$" + symbol + r"$\}$" if symbol is not None else None,
            interpolation,
            cbar_ticks=[-torch.pi, 0, torch.pi],
            cbar_ticklabels=[r"$-\pi$", r"$0$", r"$\pi$"],
        )

        axes[1].get_images()[0].set_interpolation("none")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        create_image_subplot(ax, tensor, extent, vmin, vmax, cmap, xlabel, ylabel, symbol)

    if title:
        fig.suptitle(title, y=0.95)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)

    if show:
        plt.show()

    return fig if return_fig else None


def create_image_subplot(
    ax: Any,
    tensor: Tensor,
    extent: Optional[Sequence[float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax_title: Optional[str] = None,
    interpolation: Optional[str] = None,
    cbar_ticks: Optional[Sequence[float]] = None,
    cbar_ticklabels: Optional[Sequence[str]] = None,
) -> Any:
    """
    Create an image subplot with colorbar, axis labels, and optional title.

    Args:
        ax (Any): Matplotlib axis to draw on.
        tensor (Tensor): 2D tensor to visualize.
        extent (Sequence[float], optional): Bounding box (left, right, bottom, top).
        vmin (float, optional): Minimum color scale.
        vmax (float, optional): Maximum color scale.
        cmap (str, optional): Colormap name.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        ax_title (str, optional): Title of the subplot.
        interpolation (str, optional): Interpolation type for imshow.
        cbar_ticks (Sequence[float], optional): Ticks to display on the colorbar.
        cbar_ticklabels (Sequence[str], optional): Labels for the colorbar ticks.

    Returns:
        Any: The image object returned by `imshow`.
    """
    extent_tuple = tuple(extent) if extent is not None else None
    im = ax.imshow(tensor, extent=extent_tuple, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(im, cax=cax, orientation="vertical")
    if cbar_ticks is not None:
        colorbar.set_ticks(cbar_ticks)
    if cbar_ticklabels is not None:
        colorbar.set_ticklabels(cbar_ticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(ax_title)
    return im


def animate_tensor(
    tensor: Tensor,
    title: Union[str, Sequence[str], None] = None,
    extent: Optional[Sequence[float]] = None,
    vmin: Union[float, Sequence[float], None] = None,
    vmax: Union[float, Sequence[float], None] = None,
    cmap: str = "inferno",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    symbol: Optional[str] = None,
    interpolation: Optional[str] = None,
    show: bool = True,
    func_anim_kwargs: Optional[dict] = None,
) -> FuncAnimation:
    """
    Animate a 3D tensor over time using matplotlib.

    The first dimension of the tensor is treated as time or frame index. If the tensor is complex,
    each frame is visualized as both magnitude squared and phase.

    Args:
        tensor (Tensor): A 3D tensor of shape (T, H, W).
        title (str or Sequence[str], optional): Title for each frame, or a static title.
        extent (Sequence[float], optional): Image extent in data coordinates.
        vmin (float or Sequence[float], optional): Minimum value(s) for color scaling.
        vmax (float or Sequence[float], optional): Maximum value(s) for color scaling.
        cmap (str, optional): Colormap for the plot. Defaults to "inferno".
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        symbol (str, optional): Symbol used in subplot titles.
        interpolation (str, optional): Interpolation type for imshow.
        show (bool, optional): Whether to display the animation immediately.
        func_anim_kwargs (dict, optional): Additional keyword arguments for `FuncAnimation`.

    Returns:
        FuncAnimation: The matplotlib animation object.
    """
    if tensor.ndim < 3 or not all(s == 1 for s in tensor.shape[:-3]):
        raise ValueError(f"Expected tensor to be 3D, but got shape {tensor.shape}.")

    tensor = tensor.detach().cpu().view(tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])
    num_frames = tensor.shape[0]
    is_complex = tensor.is_complex()

    def validate_sequence(arg: Union[None, str, float, Sequence], name: str):
        if isinstance(arg, torch.Tensor) or (isinstance(arg, Sequence) and not isinstance(arg, str)):
            if len(arg) != num_frames:
                raise ValueError(f"{name} must have length {num_frames}, but got {len(arg)}.")
            return arg
        return [arg] * num_frames

    titles = validate_sequence(title, "title")
    vmins = validate_sequence(vmin, "vmin")
    vmaxs = validate_sequence(vmax, "vmax")

    fig: plt.Figure = visualize_tensor(  # type: ignore[assignment]
        tensor[0],
        title=titles[0],
        extent=extent,
        vmin=vmins[0],
        vmax=vmaxs[0],
        cmap=cmap,
        xlabel=xlabel,
        ylabel=ylabel,
        symbol=symbol,
        interpolation=interpolation,
        show=False,
        return_fig=True,
    )

    axes = fig.axes
    if is_complex:
        tensor = torch.where(tensor == -0.0 - 0.0j, 0, tensor)  # Remove numerical artifacts
        ims = [axes[0].get_images()[0], axes[1].get_images()[0]]
    else:
        ims = [axes[0].get_images()[0]]

    def update(frame: int) -> None:
        if is_complex:
            ims[0].set_array(tensor[frame].abs().square())
            ims[1].set_array(tensor[frame].angle())
        else:
            ims[0].set_array(tensor[frame])
        fig.suptitle(titles[frame], y=0.95)
        ims[0].set_clim(vmins[frame], vmaxs[frame])

    anim = FuncAnimation(fig, update, frames=num_frames, **(func_anim_kwargs or {}))  # type: ignore[arg-type]

    if show:
        plt.show()

    return anim
