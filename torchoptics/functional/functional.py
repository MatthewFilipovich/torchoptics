"""Functional utilities for torchoptics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

import torch
from torch import Tensor
from torch.fft import fft2, fftfreq, ifft2
from torch.nn.functional import grid_sample

if TYPE_CHECKING:
    from ..planar_geometry import PlanarGeometry

__all__ = [
    "calculate_centroid",
    "calculate_std",
    "conv2d_fft",
    "fftfreq_grad",
    "inner2d",
    "linspace_grad",
    "meshgrid2d",
    "outer2d",
    "plane_sample",
]


def _calculate_centroid(intensity, meshgrid):
    meshgrid = torch.stack(meshgrid)
    intensity = intensity.unsqueeze(-3)
    normalized_intensity = intensity / intensity.sum((-2, -1), keepdim=True)
    centroid = (meshgrid * normalized_intensity).sum(dim=(-2, -1))
    return centroid, meshgrid, normalized_intensity


def calculate_centroid(intensity: Tensor, meshgrid: tuple[Tensor, Tensor]) -> Tensor:
    """Calculates the centroid of an intensity distribution."""
    centroid, _, _ = _calculate_centroid(intensity, meshgrid)
    return centroid


def calculate_std(intensity: Tensor, meshgrid: tuple[Tensor, Tensor]) -> Tensor:
    """Calculates the standard deviation of an intensity distribution."""
    centroid, meshgrid, normalized_intensity = _calculate_centroid(intensity, meshgrid)
    return torch.sqrt(
        ((meshgrid - centroid.unsqueeze(-1).unsqueeze(-1)) ** 2 * normalized_intensity).sum(dim=(-2, -1))
    )


def conv2d_fft(input: Tensor, weight: Tensor) -> Tensor:  # pylint: disable=redefined-builtin
    """
    Performs a 2D convolution using Fast Fourier Transforms (FFT).

    Unlike the :func:`torch.nn.functional.conv2d` function, which performs cross-correlation,
    :func:`conv2d_fft` performs a convolution operation where the kernel is flipped.

    .. note::
        It is recommended to use :attr:`torch.float64` dtype for the input and weight tensors.

    Args:
        input (torch.Tensor): Input tensor to be convolved of shape :math:`(..., iH, iW)`.
        weight (torch.Tensor): Filters of shape :math:`(..., kH, kW)`.

    Returns:
        torch.Tensor: Convolved output tensor of shape :math:`(..., oH, oW)`.
    """
    # pylint: disable=not-callable
    input_fr = fft2(input)
    output_size = (input_fr.size(-2) - weight.size(-2) + 1, input_fr.size(-1) - weight.size(-1) + 1)
    weight_fr = fft2(weight.flip(-1, -2).conj(), s=(input_fr.size(-2), input_fr.size(-1)))
    output_fr = input_fr * weight_fr.conj()
    output = ifft2(output_fr)[..., : output_size[0], : output_size[1]]
    return output


def fftfreq_grad(n: int, d: Tensor) -> Tensor:
    """
    Returns the Discrete Fourier Transform sample frequencies with gradient tracking.

    Args:
        n (int): The number of samples.
        d (torch.Tensor): The sample spacing.

    Returns:
        torch.Tensor: The sample frequencies.
    """
    return fftfreq(n, d=1.0, dtype=d.dtype, device=d.device) / d  # pylint: disable=not-callable


def inner2d(vec1: Tensor, vec2: Tensor) -> Tensor:
    """
    Computes the inner product of two 2D vectors.

    Args:
        vec1 (torch.Tensor): The first vector.
        vec2 (torch.Tensor): The second vector.

    Returns:
        torch.Tensor: The inner product.
    """
    return (vec1 * vec2.conj()).sum(dim=(-1, -2))


def linspace_grad(start: Tensor, end: Tensor, steps: int) -> Tensor:
    """
    Returns linspace values with gradient tracking.

    Args:
        start (torch.Tensor): The starting value.
        end (torch.Tensor): The ending value.
        steps (int): The number of steps.

    Returns:
        torch.Tensor: The linearly spaced values."""
    if steps == 1:
        return start
    step = (end - start) / (steps - 1)
    range_values = torch.arange(0, steps, device=start.device, dtype=start.dtype)
    return start + range_values * step


def meshgrid2d(bounds: Union[Tensor, Sequence[Tensor]], shape: Sequence) -> tuple[Tensor, Tensor]:
    """
    Returns a 2D meshgrid with gradient tracking.

    Args:
        bounds (Union[torch.Tensor, Sequence[torch.Tensor]]): The bounds of the grid.
        shape (Sequence): The shape of the grid.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The 2D meshgrid.
    """
    return torch.meshgrid(
        linspace_grad(bounds[0], bounds[1], shape[0]),
        linspace_grad(bounds[2], bounds[3], shape[1]),
        indexing="ij",
    )  # type: ignore[return-value]


def outer2d(vec1: Tensor, vec2: Tensor) -> Tensor:
    """
    Computes the outer product of two 2D vectors.

    Args:
        vec1 (torch.Tensor): The first vector.
        vec2 (torch.Tensor): The second vector.

    Returns:
        torch.Tensor: The outer product.
    """
    return vec1[..., None, None, :, :] * vec2.conj()[..., None, None]


def plane_sample(
    data: Tensor,
    data_plane: PlanarGeometry,
    interpolated_plane: PlanarGeometry,
    interpolation_mode: str,
) -> Tensor:
    """
    Interpolates data from a 2D plane onto a new plane.

    Args:
        data (torch.Tensor): The input data to interpolate.
        data_plane (PlanarGeometry): The plane containing the input data.
        interpolated_plane (PlanarGeometry): The plane to interpolate the data to.
        interpolation_mode (str, optional): The interpolation mode.

    Returns:
        torch.Tensor: The interpolated data.
    """
    # pylint: disable=too-many-locals
    data_plane_half_length = data_plane.length(use_grid_points=False) / 2
    relative_bounds = interpolated_plane.bounds(use_grid_points=True) - data_plane.offset.repeat_interleave(2)
    extent_ratio = relative_bounds / data_plane_half_length.repeat_interleave(2)

    data_reshape = (
        data.unsqueeze(0).unsqueeze(0) if data.ndim == 2 else data.flatten(end_dim=-3).unsqueeze(-3)
    )

    grid_x, grid_y = meshgrid2d(extent_ratio, interpolated_plane.shape)
    grid = torch.stack((grid_y, grid_x), dim=-1).expand(data_reshape.shape[0], -1, -1, -1)

    if torch.is_complex(data):
        real_sample = grid_sample(data_reshape.real, grid, interpolation_mode, align_corners=False)
        imag_sample = grid_sample(data_reshape.imag, grid, interpolation_mode, align_corners=False)
        interpolated_data = torch.complex(real_sample, imag_sample)
    else:
        interpolated_data = grid_sample(data_reshape, grid, interpolation_mode, align_corners=False)

    # Adjust the output shape to match the original data shape with interpolated plane dimensions.
    final_shape = data.shape[:-2] + interpolated_data.shape[-2:]
    return interpolated_data.view(final_shape)
