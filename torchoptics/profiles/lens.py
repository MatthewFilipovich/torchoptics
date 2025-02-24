"""This module defines functions to generate a thin lens profile."""

from typing import Optional

import torch

from ..config import wavelength_or_default
from ..planar_geometry import PlanarGeometry
from ..type_defs import Scalar, Vector2

__all__ = ["lens"]


def lens(
    shape: Vector2,
    focal_length: Scalar,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
    is_circular_lens: bool = True,
):
    r"""
    Generates a thin lens profile.

    The thin lens profile is defined by the following equation:

    .. math::
        \mathcal{M}(x, y) = \exp\left(-i \frac{\pi}{\lambda f} (x^2 + y^2) \right)

    where:
        - :math:`\lambda` is the wavelength of the light, and
        - :math:`f` is the focal length of the lens.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        focal_length (Scalar): Focal length of the lens.
        wavelength (Optional[Scalar]): Wavelength used for lens operation. Default: if `None`, uses a
            global default (see :meth:`torchoptics.config.set_default_wavelength()`).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
        is_circular_lens (bool): If `True`, the lens is circular and the phase profile is set to zero outside
            the lens diameter, otherwise lens is square. Default: `True`.
    """
    wavelength = wavelength_or_default(wavelength)

    planar_geometry = PlanarGeometry(shape, spacing=spacing, offset=offset)
    x, y = planar_geometry.meshgrid()
    radial_dist = torch.sqrt(x**2 + y**2)
    phase_profile = torch.exp(-1j * torch.pi / (wavelength * focal_length) * radial_dist**2)

    if is_circular_lens:
        lens_diameter = min(planar_geometry.length(use_grid_points=True))
        mask = radial_dist > lens_diameter / 2
        phase_profile[mask] = 0
    return phase_profile
