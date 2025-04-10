"""This module defines functions to generate a quadratic phase profile."""

from typing import Optional

import torch

from ..config import wavelength_or_default
from ..planar_grid import PlanarGrid
from ..type_defs import Scalar, Vector2
from .shapes import circle

__all__ = ["lens"]


def lens(
    shape: Vector2,
    focal_length: Scalar,
    radius: Scalar,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
):
    r"""
    Generates a quadratic phase profile, which can be used to represent a thin lens.

    The quadratic phase profile is defined by the following equation:

    .. math::
        \mathcal{M}(x, y) = \exp\left(-i \frac{\pi}{\lambda f} (x^2 + y^2) \right)

    where:
        - :math:`\lambda` is the wavelength of the light, and
        - :math:`f` is the focal length.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        focal_length (Scalar): Focal length in quadratic phase shift.
        wavelength (Optional[Scalar]): Wavelength in quadratic phase shift. Default: if `None`, uses a
            global default (see :meth:`torchoptics.config.set_default_wavelength()`).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """
    wavelength = wavelength_or_default(wavelength)

    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    x, y = planar_grid.meshgrid()
    radial_square = x**2 + y**2

    quadratic_phase = torch.exp(-1j * torch.pi / (wavelength * focal_length) * radial_square)
    circle_profile = circle(shape, radius, spacing=spacing, offset=offset)
    phase_profile = quadratic_phase * circle_profile

    return phase_profile
