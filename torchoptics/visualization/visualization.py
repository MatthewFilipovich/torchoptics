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
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    symbol: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
    **imshow_kwargs,
) -> Optional[plt.Figure]:
    """
    Visualize a 2D real or complex-valued tensor using matplotlib.

    If the tensor is complex, two subplots are shown: one for the magnitude squared and one for the phase.

    Args:
        tensor (Tensor): A 2D tensor of shape (H, W).
        title (str, optional): Title for the figure.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        symbol (str, optional): Symbol used in subplot titles for LaTeX rendering.
        show (bool, optional): Whether to call `plt.show()`. Defaults to True.
        return_fig (bool, optional): If True, returns the matplotlib Figure.
        **imshow_kwargs: Additional keyword arguments passed directly to `matplotlib.pyplot.imshow()`, such as
            `cmap`, `vmin`, `vmax`, `interpolation`, etc.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure if `return_fig` is True, else None.
    """
    if tensor.ndim < 2 or not all(s == 1 for s in tensor.shape[:-2]):
        raise ValueError(f"Expected tensor to be 2D, but got shape {tensor.shape}.")

    tensor = tensor.detach().cpu().view(tensor.shape[-2], tensor.shape[-1])

    if tensor.is_complex():  # Creates two subplots for complex tensors
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        tensor = torch.where(tensor == -0.0 - 0.0j, 0, tensor)

        create_image_subplot(  # Plot magnitude squared
            ax=axes[0],
            tensor=tensor.abs().square(),
            xlabel=xlabel,
            ylabel=ylabel,
            ax_title=rf"$|${symbol}$|^2$" if symbol else None,
            **imshow_kwargs,
        )

        imshow_phase_kwargs = {  # Override imshow kwargs for phase plot
            **imshow_kwargs,
            "vmin": -torch.pi,
            "vmax": torch.pi,
            "cmap": "twilight_shifted",
            "norm": None,
            "interpolation": "none",
        }

        create_image_subplot(  # Plot phase
            ax=axes[1],
            tensor=tensor.angle(),
            xlabel=xlabel,
            ylabel=ylabel,
            ax_title=r"$\arg \{$" + symbol + r"$\}$" if symbol is not None else None,
            cbar_ticks=[-torch.pi, 0, torch.pi],
            cbar_ticklabels=[r"$-\pi$", r"$0$", r"$\pi$"],
            **imshow_phase_kwargs,
        )

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

    else:  # Plots a single subplot for real tensors
        fig, ax = plt.subplots(figsize=(5, 5))
        create_image_subplot(
            ax=ax,
            tensor=tensor,
            xlabel=xlabel,
            ylabel=ylabel,
            ax_title=symbol,
            **imshow_kwargs,
        )

    if title:
        fig.suptitle(title, y=0.95)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)

    if show:
        plt.show()

    return fig if return_fig else None


def animate_tensor(
    tensor: Tensor,
    title: Union[str, Sequence[str], None] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    symbol: Optional[str] = None,
    show: bool = True,
    func_anim_kwargs: Optional[dict] = None,
    **imshow_kwargs,
) -> FuncAnimation:
    """
    Animate a 3D tensor over time using matplotlib.

    The first dimension of the tensor is treated as time or frame index. If the tensor is complex,
    each frame is visualized as both magnitude squared and phase.

    Args:
        tensor (Tensor): A 3D tensor of shape (T, H, W).
        title (str or Sequence[str], optional): Title for each frame, or a static title.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        symbol (str, optional): Symbol used in subplot titles for LaTeX rendering.
        show (bool, optional): Whether to call `plt.show()`. Defaults to True.
        func_anim_kwargs (dict, optional): Additional keyword arguments for `FuncAnimation`.
        **imshow_kwargs: Additional keyword arguments passed directly to `matplotlib.pyplot.imshow()`, such as
            `cmap`, `vmin`, `vmax`, `interpolation`, etc.

    Returns:
        FuncAnimation: The matplotlib animation object.
    """
    if tensor.ndim < 3 or not all(s == 1 for s in tensor.shape[:-3]):
        raise ValueError(f"Expected tensor to be 3D, but got shape {tensor.shape}.")

    tensor = tensor.detach().cpu().view(tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])
    num_frames = tensor.shape[0]
    is_complex = tensor.is_complex()

    titles = [title] * num_frames if isinstance(title, str) or title is None else list(title)

    if len(titles) != num_frames:
        raise ValueError(f"`title` must have length {num_frames}, but got {len(titles)}.")

    fig: plt.Figure = visualize_tensor(  # type: ignore[assignment]
        tensor[0],
        title=titles[0],
        xlabel=xlabel,
        ylabel=ylabel,
        symbol=symbol,
        show=False,
        return_fig=True,
        **imshow_kwargs,
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

        if titles[frame]:
            fig.suptitle(titles[frame], y=0.95)  # type: ignore[arg-type]

    anim = FuncAnimation(fig, update, frames=num_frames, **(func_anim_kwargs or {}))  # type: ignore[arg-type]

    if show:
        plt.show()

    return anim


def create_image_subplot(
    ax: Any,
    tensor: Tensor,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax_title: Optional[str] = None,
    cbar_ticks: Optional[Sequence[float]] = None,
    cbar_ticklabels: Optional[Sequence[str]] = None,
    **imshow_kwargs,
) -> Any:
    """
    Create an image subplot with colorbar, axis labels, and optional title.

    Args:
        ax (Any): Matplotlib axis to draw on.
        tensor (Tensor): 2D tensor to visualize.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        ax_title (str, optional): Title of the subplot.
        cbar_ticks (Sequence[float], optional): Ticks to display on the colorbar.
        cbar_ticklabels (Sequence[str], optional): Labels for the colorbar ticks.
        **imshow_kwargs: Additional keyword arguments passed directly to `matplotlib.pyplot.imshow()`, such as
            `cmap`, `vmin`, `vmax`, `interpolation`, etc.

    Returns:
        Any: The image object returned by `imshow`.
    """
    imshow_kwargs.setdefault("cmap", "inferno")

    im = ax.imshow(tensor, **imshow_kwargs)
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
