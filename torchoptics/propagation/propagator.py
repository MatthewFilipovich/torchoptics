"""This module defines functions to propagate Field objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ..functional import plane_sample
from ..planar_geometry import PlanarGeometry
from ..type_defs import Scalar, Vector2
from ..utils import copy
from .angular_spectrum_method import asm_propagation
from .direct_integration_method import calculate_grid_bounds, dim_propagation

if TYPE_CHECKING:
    from ..fields import Field

__all__ = [
    "propagator",
    "get_propagation_plane",
    "is_asm",
    "calculate_critical_propagation_distance",
]

VALID_PROPAGATION_METHODS = {"AUTO", "AUTO_FRESNEL", "ASM", "ASM_FRESNEL", "DIM", "DIM_FRESNEL"}
VALID_INTERPOLATION_MODES = {"none", "bilinear", "bicubic", "nearest"}


def propagator(
    field: Field,
    shape: Vector2,
    z: Scalar,
    spacing: Optional[Vector2],
    offset: Optional[Vector2],
    propagation_method: str,
    asm_pad_factor: Vector2,
    interpolation_mode: str,
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
    validate_propagation_method(propagation_method)
    validate_interpolation_mode(interpolation_mode)

    output_plane = PlanarGeometry(shape, z, spacing, offset).to(field.data.device)

    if output_plane.z != field.z:  # Propagate to output plane z
        propagation_plane = get_propagation_plane(field, output_plane)
        if is_asm(field, propagation_plane, propagation_method):
            field = asm_propagation(field, propagation_plane, propagation_method, asm_pad_factor)
        else:
            field = dim_propagation(field, propagation_plane, propagation_method)

    if not output_plane.is_same_geometry(field):  # Interpolate to output plane geometry
        transformed_data = plane_sample(field.data, field, output_plane, interpolation_mode)
        field = copy(field, data=transformed_data, spacing=output_plane.spacing, offset=output_plane.offset)

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


def is_asm(field: Field, propagation_plane: PlanarGeometry, propagation_method: str):
    """Returns whether propagation using ASM should be used.

    Returns `True` if :attr:`field.propagation_method` is `"ASM"` or `"ASM_FRESNEL"`.

    Returns `False` if :attr:`field.propagation_method` is `"DIM"` or `"DIM_FRESNEL"`.

    If :attr:`field.propagation_method` is `"auto"`, the propagation method is determined based on the
    condition set in :func:`calculate_critical_propagation_distance`. Returns `True` if at least one of the
    two planar dimensions meets the condition; otherwise, returns `False`.
    """
    if propagation_method.upper() in ("DIM", "DIM_FRESNEL"):
        return False
    if propagation_method.upper() in ("ASM", "ASM_FRESNEL"):
        return True

    # Auto: Determine propagation method based on critical propagation distance
    critical_z = calculate_critical_propagation_distance(field, propagation_plane)
    z = (propagation_plane.z - field.z).abs()
    return torch.any(z < critical_z)


def calculate_critical_propagation_distance(field: Field, propagation_plane: PlanarGeometry) -> torch.Tensor:
    r"""
    Calculates the critical propagation distance for determining the propagation method.

    The minimum distance is calculated using the criteria in Eq. A.17 from D. Voelz's textbook "Computational
    Fourier Optics: A MATLAB Tutorial" (2011):

    .. math::
        z_c = \frac{2 |x_\mathrm{max}| \Delta}{\lambda},

    where:

    - :math:`z_c` is the critical propagation distance.
    - :math:`x_\mathrm{max}` is the maximum distance between grid points of the field and propagation planes.
    - :math:`\Delta` is the spacing of the field.
    - :math:`\lambda` is the wavelength of the field.

    The returned value is a tensor of shape (2,) containing the critical distances in both planar dimensions.
    """
    grid_bounds_abs = calculate_grid_bounds(field, propagation_plane).abs()
    max_distance = torch.stack([max(grid_bounds_abs[:2]), max(grid_bounds_abs[2:])])
    return 2 * max_distance * field.spacing / field.wavelength


def validate_propagation_method(value: str) -> None:
    """Validate the propagation method."""
    if not isinstance(value, str):
        raise TypeError(f"Expected propagation_method to be a string, but got {type(value).__name__}.")

    if value.upper() not in VALID_PROPAGATION_METHODS:
        raise ValueError(
            f"Expected propagation_method to be one of {VALID_PROPAGATION_METHODS}, but got {value}."
        )


def validate_interpolation_mode(value: str) -> None:
    """Validate the interpolation mode."""
    if not isinstance(value, str):
        raise TypeError(f"Expected interpolation_mode to be a string, but got {type(value).__name__}.")
    if value.lower() not in VALID_INTERPOLATION_MODES:
        raise ValueError(
            f"Expected interpolation_mode to be one of {VALID_INTERPOLATION_MODES}, but got {value}."
        )
