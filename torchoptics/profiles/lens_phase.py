"""This module defines functions to generate quadratic phase lens profiles."""

from math import cos, sin
from typing import Optional

import torch

from ..config import wavelength_or_default
from ..planar_grid import PlanarGrid
from ..type_defs import Scalar, Vector2
from ..utils import initialize_tensor

__all__ = ["lens_phase", "cylindrical_lens_phase"]


def lens_phase(
    shape: Vector2,
    focal_length: Scalar,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
):
    r"""
    Generates a quadratic phase lens profile, which can be used to represent a thin lens.

    The quadratic phase profile is defined by the following equation:

    .. math::
        \psi(x, y) = -i \frac{\pi}{\lambda f} (x^2 + y^2)

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        focal_length (Scalar): Focal length of the lens.
        wavelength (Optional[Scalar]): Wavelength used for lens operation.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.
    """
    wavelength = wavelength_or_default(wavelength)
    focal_length = initialize_tensor("focal_length", focal_length, is_scalar=True)

    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    x, y = planar_grid.meshgrid()
    radial_square = x**2 + y**2

    return -torch.pi / (wavelength * focal_length) * radial_square


def cylindrical_lens_phase(
    shape: Vector2,
    focal_length: Scalar,
    theta: Scalar = 0.0,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
):
    r"""
    Generates a cylindrical lens profile with a quadratic phase in a specified direction.

    The quadratic phase profile is:

    .. math::
        \psi(x, y) = -i \frac{\pi}{\lambda f} (x_\theta)^2

    where :math:`x_\theta = x \cos\theta + y \sin\theta`.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        focal_length (Scalar): Focal length along the axis of the lens.
        theta (Scalar): Orientation angle of the cylindrical axis in radians.
        wavelength (Optional[Scalar]): Wavelength used for lens operation.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.
    """
    wavelength = wavelength_or_default(wavelength)
    focal_length = initialize_tensor("focal_length", focal_length, is_scalar=True)
    theta = initialize_tensor("theta", theta, is_scalar=True)

    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    x, y = planar_grid.meshgrid()
    x_theta = x * cos(theta) + y * sin(theta)

    return -torch.pi / (wavelength * focal_length) * x_theta**2
