"""This module defines functions to propagate Field objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ..functional import plane_sample
from ..planar_geometry import PlanarGeometry
from ..type_defs import Scalar, Vector2
from .angular_spectrum_method import asm_propagation
from .direct_integration_method import dim_propagation

if TYPE_CHECKING:
    from ..fields import Field

__all__ = [
    "propagator",
    "get_propagation_plane",
    "is_dim_propagation",
    "calculate_min_dim_propagation_distance",
]


def propagator(
    field: Field,
    shape: Vector2,
    z: Scalar,
    spacing: Optional[Vector2],
    offset: Optional[Vector2],
) -> Field:
    """
    Propagates the field through free-space to a plane defined by the input parameters.

    First, the field is propagated to the plane determined by :meth:`get_propagation_plane()`. This propagated
    field is then interpolated using :func:`torchoptics.functional.plane_sample()` to match the geometry of
    the output plane.

    Args:
        field (Field): Input field.
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.

    Returns:
        Field: Output field after propagating to the plane.
    """
    output_plane = PlanarGeometry(shape, z, spacing, offset).to(field.data.device)

    if output_plane.z != field.z:  # Propagate to output plane z
        propagation_plane = get_propagation_plane(field, output_plane)
        propagation_func = (
            dim_propagation if is_dim_propagation(field, propagation_plane) else asm_propagation
        )
        field = propagation_func(field, propagation_plane)

    if not output_plane.is_same_geometry(field):  # Interpolate to output plane geometry
        transformed_data = plane_sample(field.data, field, output_plane, field.interpolation_mode)
        field = field.copy(data=transformed_data, spacing=output_plane.spacing, offset=output_plane.offset)

    return field


def get_propagation_plane(field: Field, output_plane: PlanarGeometry) -> PlanarGeometry:
    r"""
    Creates a propagation plane that is equal to or slightly larger than the specified output plane.

    The propagation plane adopts the same ``spacing`` as the ``field``, and retains the same ``z`` and
    ``offset`` values as the ``output_plane``.

    The length of the propagation plane must satisfy the inequality:

    .. math::
        \text{propagation plane length} \geq \text{output plane length}.

    This can be expressed as:

    .. math::
        (N_{{\text{prop}}} - 1) \cdot \Delta_{{\text{prop}}} \geq (N_{{\text{out}}} - 1) \cdot
        \Delta_{{\text{out}}}.

    Therefore, the number of grid points in the propagation plane, :math:`N_{{\text{prop}}}`, must be:

    .. math::
        N_{{\text{prop}}} \geq \left [ \frac{{(N_{{\text{out}}} - 1)
        \cdot \Delta_{{\text{out}}}}}{{\Delta_{{\text{prop}}}}} \right ] + 1,

    where:

    - :math:`N_{{\text{prop}}}` is the number of grid points (``shape``) in the propagation plane.
    - :math:`N_{{\text{out}}}` is the number of grid points (``shape``) in the output plane.
    - :math:`\Delta_{{\text{prop}}}` is the spacing in the propagation plane.
    - :math:`\Delta_{{\text{out}}}` is the spacing in the output plane.
    """
    spacing_ratio = output_plane.spacing / field.spacing
    output_plane_shape = torch.tensor(output_plane.shape, device=spacing_ratio.device)
    propagation_shape = torch.ceil((output_plane_shape - 1) * spacing_ratio).int() + 1
    return PlanarGeometry(propagation_shape, output_plane.z, field.spacing, output_plane.offset)


def is_dim_propagation(field: Field, propagation_plane: PlanarGeometry) -> bool:
    """
    Returns whether propagation using DIM should be used.

    Returns `True` if :attr:`field.propagation_method` is `"DIM"` or `"DIM_FRESNEL"`.

    Returns `False` if :attr:`field.propagation_method` is `"ASM"` or `"ASM_FRESNEL"`.

    If :attr:`field.propagation_method` is `"auto"`, the propagation method is determined based on the
    condition set in :func:`calculate_min_dim_propagation_distance`.
    """
    if field.propagation_method in ("DIM", "DIM_FRESNEL"):
        return True
    if field.propagation_method in ("ASM", "ASM_FRESNEL"):
        return False
    abs_propagation_distance = abs(propagation_plane.z - field.z)
    return calculate_min_dim_propagation_distance(field) < abs_propagation_distance


def calculate_min_dim_propagation_distance(field: Field) -> float:
    r"""
    Calculates the minimum distance required for accurate simulation using DIM.

    The minimum distance is calculated using the criteria from D. Voelz's textbook "Computational Fourier
    Optics: A MATLAB Tutorial" (2011):

    .. math::
        z \geq L \Delta / \lambda,

    where:

    - :math:`z` is the propagation distance.
    - :math:`L` is the maximum length of the field.
    - :math:`\Delta` is the spacing of the field.
    - :math:`\lambda` is the wavelength of the field.
    """
    return float((field.length() * field.spacing).max() / field.wavelength)
