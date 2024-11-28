"""This module defines functions for field propagation using the direct integration method (DIM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..functional import conv2d_fft, meshgrid2d
from ..planar_geometry import PlanarGeometry

if TYPE_CHECKING:
    from ..fields import Field

__all__ = ["dim_propagation"]


def dim_propagation(field: Field, propagation_plane: PlanarGeometry) -> Field:
    """
    Propagates the field to a plane using the direct integration method (DIM).

    Args:
        field (Field): Input field.
        propagation_plane (PlanarGeometry): Plane to which the field is propagated.

    Returns:
        Field: Output field after propagation.
    """
    x, y = calculate_meshgrid(field, propagation_plane)
    impulse_response = calculate_impulse_response(field, propagation_plane, x, y)
    propagated_data = conv2d_fft(impulse_response, field.data)
    return field.copy(data=propagated_data, z=propagation_plane.z, offset=propagation_plane.offset)


def calculate_meshgrid(field: Field, propagation_plane: PlanarGeometry) -> tuple[Tensor, Tensor]:
    """Calculate the meshgrid for the impulse response calculation."""
    field_bounds = field.bounds(use_grid_points=True)
    propagation_plane_bounds = propagation_plane.bounds(use_grid_points=True)

    grid_bounds = [
        propagation_plane_bounds[0] - field_bounds[1],
        propagation_plane_bounds[1] - field_bounds[0],
        propagation_plane_bounds[2] - field_bounds[3],
        propagation_plane_bounds[3] - field_bounds[2],
    ]
    grid_shape = [field.shape[i] + propagation_plane.shape[i] - 1 for i in range(2)]
    return meshgrid2d(grid_bounds, grid_shape)


def calculate_impulse_response(
    field: Field, propagation_plane: PlanarGeometry, x: Tensor, y: Tensor
) -> Tensor:
    """Calculate the impulse response for DIM propagation."""
    propagation_distance = propagation_plane.z - field.z
    r_squared = x**2 + y**2 + propagation_distance**2
    r = torch.sqrt(r_squared) if propagation_distance >= 0 else -torch.sqrt(r_squared)
    k = 2 * torch.pi / field.wavelength
    if field.propagation_method in {"DIM_FRESNEL", "AUTO_FRESNEL"}:
        impulse_response = (
            (torch.exp(1j * k * propagation_distance) / (1j * field.wavelength * propagation_distance))
            * torch.exp(1j * k / (2 * propagation_distance) * (x**2 + y**2))
            * field.cell_area()
        )
    else:  # DIM using RS equation
        impulse_response = (
            1 / (2 * torch.pi) * (1 / r - 1j * k) * torch.exp(1j * k * r) * propagation_distance / r_squared
        ) * field.cell_area()
    return impulse_response
