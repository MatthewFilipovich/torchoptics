"""Field propagation using the angular spectrum method (ASM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.nn.functional import pad

from ..functional import fftfreq_grad
from ..utils import initialize_tensor

if TYPE_CHECKING:
    from ..fields import Field
    from ..planar_grid import PlanarGrid
    from ..types import Vector2


def asm_propagation(
    field: Field,
    propagation_plane: PlanarGrid,
    propagation_method: str,
    asm_pad: Vector2 | None,
) -> Field:
    """Propagate the field to a plane using the angular spectrum method (ASM).

    Args:
        field (Field): Input field.
        propagation_plane (PlanarGrid): Plane to which the field is propagated.
        propagation_method (str): Propagation method to use.
        asm_pad (Vector2 | None): Padding size for ASM propagation.

    Returns:
        Field: Output field after propagation.

    """
    if asm_pad is None:  # Default padding is 2x the input field size in each dimension
        asm_pad = [2 * field.shape[0], 2 * field.shape[1]]
    asm_pad = initialize_tensor("asm_pad", asm_pad, is_vector2=True, is_integer=True, is_non_negative=True)
    propagation_distance = propagation_plane.z - field.z
    transfer_function = calculate_transfer_function(field, propagation_distance, asm_pad, propagation_method)
    propagated_data = apply_transfer_function(transfer_function, field, asm_pad)
    propagated_field = field.copy(data=propagated_data, z=propagation_plane.z)
    validate_bounds(propagated_field, propagation_plane, asm_pad)
    return propagated_field


def calculate_transfer_function(
    field: Field,
    propagation_distance: Tensor,
    asm_pad: Tensor,
    propagation_method: str,
) -> Tensor:
    """Calculate the transfer function for ASM propagation."""
    padded_input_shape = torch.tensor(field.shape) + 2 * asm_pad
    freq_x, freq_y = (
        fftshift(fftfreq_grad(n, d)) for n, d in zip(padded_input_shape, field.spacing, strict=False)
    )
    kx, ky = torch.meshgrid(freq_x * 2 * torch.pi, freq_y * 2 * torch.pi, indexing="ij")
    k = 2 * torch.pi / field.wavelength

    if propagation_method.upper() in ("ASM", "AUTO"):  # Unnamed default uses Rayleigh-Sommerfeld (RS)
        kz_squared = (k**2 - kx**2 - ky**2) + 0j  # kz_squared is complex for sqrt calculation
        kz = torch.sqrt(kz_squared)  # kz is imaginary for evanescent waves where kz^2 < 0
        return torch.exp(1j * kz * propagation_distance)

    # Explicit _FRESNEL variants use the Fresnel approximation.
    return torch.exp(1j * k * propagation_distance) * torch.exp(
        -1j * field.wavelength * propagation_distance * (kx**2 + ky**2) / (4 * torch.pi),
    )


def apply_transfer_function(transfer_function: Tensor, field: Field, asm_pad: Tensor) -> Tensor:
    """Apply the transfer function to the field for ASM propagation."""
    pad_x, pad_y = int(asm_pad[0]), int(asm_pad[1])
    data = pad(field.data, (pad_y, pad_y, pad_x, pad_x), mode="constant", value=0)
    data = fftshift(fft2(data))
    data = data * transfer_function
    return ifft2(ifftshift(data))


def validate_bounds(propagated_field: Field, target_plane: PlanarGrid, asm_pad: Vector2) -> None:
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

        msg = (
            f"Propagation plane bounds {formatted_target_bounds} are outside padded field bounds "
            f"{formatted_propagated_bounds}.\nIncrease asm_pad ({asm_pad}) "
            f"or adjust propagation plane offset ({formatted_target_offset})."
        )
        raise ValueError(msg)
