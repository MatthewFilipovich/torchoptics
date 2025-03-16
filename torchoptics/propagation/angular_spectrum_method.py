"""This module defines functions for field propagation using the angular spectrum method (ASM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.nn.functional import pad

from ..functional import fftfreq_grad
from ..planar_geometry import PlanarGeometry
from ..type_defs import Vector2
from ..utils import copy, initialize_tensor

if TYPE_CHECKING:
    from ..fields import Field

__all__ = ["asm_propagation"]


def asm_propagation(
    field: Field, propagation_plane: PlanarGeometry, propagation_method: str, asm_pad_factor: Vector2
) -> Field:
    """
    Propagates the field to a plane using the angular spectrum method (ASM).

    Args:
        field (Field): Input field.
        propagation_plane (PlanarGeometry): Plane to which the field is propagated.

    Returns:
        Field: Output field after propagation.
    """
    asm_pad_factor = initialize_tensor(
        "asm_pad_factor", asm_pad_factor, is_vector2=True, is_integer=True, is_non_negative=True
    )
    propagation_distance = propagation_plane.z - field.z
    transfer_function = calculate_transfer_function(
        field, propagation_distance, asm_pad_factor, propagation_method
    )
    propagated_data = apply_transfer_function(transfer_function, field, asm_pad_factor)
    propagated_field = copy(field, data=propagated_data, z=propagation_plane.z)
    validate_bounds(propagated_field, propagation_plane, asm_pad_factor)
    return propagated_field


def calculate_transfer_function(
    field: Field, propagation_distance: Tensor, asm_pad_factor: Tensor, propagation_method: str
) -> Tensor:
    """Calculate the transfer function for ASM propagation."""
    # pylint: disable=not-callable
    padded_input_shape = torch.tensor(field.shape) * (2 * asm_pad_factor + 1)
    freq_x, freq_y = (fftshift(fftfreq_grad(n, d)) for n, d in zip(padded_input_shape, field.spacing))
    kx, ky = torch.meshgrid(freq_x * 2 * torch.pi, freq_y * 2 * torch.pi, indexing="ij")
    k = 2 * torch.pi / field.wavelength
    kz_squared = (k**2 - kx**2 - ky**2).to(torch.cdouble)  # Ensure kz_squared is complex for sqrt calculation
    kz = torch.sqrt(kz_squared)  # kz is imaginary for evanescent waves where kz^2 < 0

    if propagation_method in {"ASM_FRESNEL", "AUTO_FRESNEL"}:
        return torch.exp(1j * k * propagation_distance) * torch.exp(
            -1j * field.wavelength * propagation_distance * (kx**2 + ky**2) / (4 * torch.pi)
        )
    return torch.exp(1j * kz * propagation_distance)  # ASM using RS equation


def apply_transfer_function(transfer_function: Tensor, field: Field, asm_pad_factor: Tensor) -> Tensor:
    """Apply the transfer function to the field for ASM propagation."""
    # pylint: disable=not-callable
    pad_x, pad_y = [int(asm_pad_factor[i] * field.shape[i]) for i in range(2)]
    data = pad(field.data, (pad_y, pad_y, pad_x, pad_x), mode="constant", value=0)
    data = fftshift(fft2(data))
    data = data * transfer_function
    data = ifft2(ifftshift(data))
    return data


def validate_bounds(propagated_field: Field, target_plane: PlanarGeometry, asm_pad_factor: Vector2) -> None:
    """Validate that the propagated field bounds contain the target plane bounds."""
    target_plane_bounds = target_plane.bounds(use_grid_points=True)
    propagated_field_bounds = propagated_field.bounds(use_grid_points=True)
    if (
        target_plane_bounds[0] < propagated_field_bounds[0]
        or target_plane_bounds[1] > propagated_field_bounds[1]
        or target_plane_bounds[2] < propagated_field_bounds[2]
        or target_plane_bounds[3] > propagated_field_bounds[3]
    ):
        formatted_target_bounds = [f"{val:.2e}" for val in target_plane_bounds]
        formatted_propagated_bounds = [f"{val:.2e}" for val in propagated_field_bounds]
        formatted_target_offset = [f"{val:.2e}" for val in target_plane.offset]

        raise ValueError(
            f"Propagation plane bounds {formatted_target_bounds} are outside padded field bounds "
            f"{formatted_propagated_bounds}.\nIncrease asm_pad_factor ({asm_pad_factor}) "
            f"or adjust propagation plane offset ({formatted_target_offset})."
        )
