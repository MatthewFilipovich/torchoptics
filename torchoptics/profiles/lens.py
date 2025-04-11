"""This module defines functions to generate quadratic phase lens profiles."""

from math import cos, sin
from typing import Optional

import torch

from ..config import wavelength_or_default
from ..planar_grid import PlanarGrid
from ..type_defs import Scalar, Vector2
from ..utils import initialize_tensor
from .shapes import circle

__all__ = ["lens", "cylindrical_lens"]


def lens(
    shape: Vector2,
    focal_length: Scalar,
    radius: Scalar,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
):
    r"""
    Generates a quadratic phase lens profile, which can be used to represent a thin lens.

    The quadratic phase profile is defined by the following equation:

    .. math::
        \mathcal{M}(x, y) = \exp\left(-i \frac{\pi}{\lambda f} (x^2 + y^2) \right)

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        focal_length (Scalar): Focal length of the lens.
        radius (Scalar): Radius of the circular aperture.
        wavelength (Optional[Scalar]): Wavelength used for lens operation.
        spacing (Optional[Vector2]): Grid spacing.
        offset (Optional[Vector2]): Grid offset.
    """
    wavelength = wavelength_or_default(wavelength)
    focal_length = initialize_tensor("focal_length", focal_length, is_scalar=True)
    radius = initialize_tensor("radius", radius, is_scalar=True, is_positive=True)

    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    x, y = planar_grid.meshgrid()
    radial_square = x**2 + y**2

    quadratic_phase = torch.exp(-1j * torch.pi / (wavelength * focal_length) * radial_square)
    circle_profile = circle(shape, radius, spacing=spacing, offset=offset)
    return quadratic_phase * circle_profile


def cylindrical_lens(
    shape: Vector2,
    focal_length: Scalar,
    radius: Scalar,
    theta: Scalar = 0.0,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
):
    r"""
    Generates a cylindrical lens profile with a quadratic phase in a specified direction.

    The quadratic phase profile is:

    .. math::
        \mathcal{M}(x, y) = \exp\left(-i \frac{\pi}{\lambda f} (x_\theta)^2 \right)

    where :math:`x_\theta = x \cos\theta + y \sin\theta`.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        focal_length (Scalar): Focal length along the axis of the lens.
        radius (Scalar): Radius of the circular aperture.
        theta (Scalar): Orientation angle of the cylindrical axis in radians.
        wavelength (Optional[Scalar]): Wavelength used for lens operation.
        spacing (Optional[Vector2]): Grid spacing.
        offset (Optional[Vector2]): Grid offset.
    """
    wavelength = wavelength_or_default(wavelength)
    focal_length = initialize_tensor("focal_length", focal_length, is_scalar=True)
    radius = initialize_tensor("radius", radius, is_scalar=True, is_positive=True)
    theta = initialize_tensor("theta", theta, is_scalar=True)

    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    x, y = planar_grid.meshgrid()
    x_theta = x * cos(theta) + y * sin(theta)

    phase = torch.exp(-1j * torch.pi / (wavelength * focal_length) * x_theta**2)
    aperture = circle(shape, radius, spacing=spacing, offset=offset)

    return phase * aperture
