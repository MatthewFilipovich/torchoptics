"""This module defines a function to generate the bessel beam profile."""

import math
from typing import Optional

import torch
from torch import Tensor

from ..config import get_default_wavelength
from ..planar_geometry import PlanarGeometry
from ..type_defs import Scalar, Vector2

__all__ = ["bessel"]


def bessel(
    shape: Vector2,
    cone_angle: float,
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
        cone_angle (float): The cone angle in radians.
        wavelength (Scalar, optional): The wavelength of the beam. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.

    Returns:
        Tensor: The generated zeroth-order Bessel beam profile.

    """
    wavelength = wavelength if wavelength is not None else get_default_wavelength()

    # Calculate the wave number k and its radial component
    k = 2 * torch.pi / wavelength  # type: ignore
    k_r = k * math.sin(cone_angle)

    # Generate the planar grid
    x, y = PlanarGeometry(shape, spacing=spacing, offset=offset).meshgrid()
    r = torch.sqrt(x**2 + y**2)

    # Calculate the zeroth-order Bessel beam
    return torch.special.bessel_j0(k_r * r)  # pylint: disable=not-callable
