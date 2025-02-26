"""This module defines a function to generate the bessel beam profile."""

from typing import Optional

import torch
from torch import Tensor

from ..config import wavelength_or_default
from ..planar_geometry import PlanarGeometry
from ..type_defs import Scalar, Vector2
from ..utils import initialize_tensor

__all__ = ["bessel"]


def bessel(
    shape: Vector2,
    cone_angle: Scalar,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
) -> Tensor:
    r"""
    Generates a zeroth-order Bessel beam.

    The zeroth-order Bessel beam is defined by the following equation:

    .. math::
        \psi(r) = J_0(k \, r \sin(\theta)),

    where:

    - :math:`J_0` is the zeroth-order Bessel function of the first kind,
    - :math:`\theta` is the cone angle of the Bessel beam, and
    - :math:`k` is the wave number, :math:`k = 2\pi / \lambda`.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        cone_angle (Scalar): The cone angle in radians.
        wavelength (Scalar, optional): The wavelength of the beam. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.

    Returns:
        Tensor: The generated zeroth-order Bessel beam profile.

    """
    wavelength = wavelength_or_default(wavelength)
    cone_angle = initialize_tensor("cone_angle", cone_angle, is_scalar=True)

    k = 2 * torch.pi / wavelength
    k_r = k * torch.sin(cone_angle)

    x, y = PlanarGeometry(shape, spacing=spacing, offset=offset).meshgrid()
    r = torch.sqrt(x**2 + y**2)

    # Calculate the zeroth-order Bessel beam
    return torch.special.bessel_j0(k_r * r)  # pylint: disable=not-callable
