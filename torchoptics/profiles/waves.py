"""This module defines functions to generate the phase arguments of spherical and plane waves."""

from typing import Optional

import torch

from ..config import wavelength_or_default
from ..planar_grid import PlanarGrid
from ..type_defs import Scalar, Vector2
from ..utils import initialize_tensor

__all__ = ["plane_wave_phase", "spherical_wave_phase"]


def plane_wave_phase(
    shape: Vector2,
    theta: Scalar = 0.0,
    phi: Scalar = 0.0,
    z: Scalar = 0.0,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
):
    r"""
    Computes the phase argument of a plane wave with arbitrary propagation direction.

    The phase argument is defined as:

    .. math::
        \psi(x, y, z) = k_x x + k_y y + k_z z

    where:
        - :math:`k_x = \frac{2\pi}{\lambda} \sin\theta \cos\phi`
        - :math:`k_y = \frac{2\pi}{\lambda} \sin\theta \sin\phi`
        - :math:`k_z = \frac{2\pi}{\lambda} \cos\theta`

    Args:
        shape (Vector2): Number of grid points along planar dimensions.
        theta (float): Polar angle from the z-axis, in radians.
        phi (float): Azimuthal angle in the x-y plane, in radians.
        z (Scalar): Axial location at which to evaluate the phase. Default: 0.
        wavelength (Optional[Scalar]): Wavelength of the wave. If None, uses global default.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the wave. Default: `(0, 0)`.

    Returns:
        torch.Tensor: Real-valued 2D phase argument of the plane wave.
    """
    wavelength = wavelength_or_default(wavelength)
    theta = initialize_tensor("theta", theta, is_scalar=True)
    phi = initialize_tensor("phi", phi, is_scalar=True)
    z = initialize_tensor("z", z, is_scalar=True)

    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    x, y = planar_grid.meshgrid()

    k0 = 2 * torch.pi / wavelength
    kx = k0 * torch.sin(theta) * torch.cos(phi)
    ky = k0 * torch.sin(theta) * torch.sin(phi)
    kz = k0 * torch.cos(theta)

    return kx * x + ky * y + kz * z


def spherical_wave_phase(
    shape: Vector2,
    z: Scalar,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
):
    r"""
    Computes the phase argument of a spherical wave originating from a point source.

    The phase argument is defined as:

    .. math::
        \psi(x, y, z) = k \cdot r, \quad
        r = \sqrt{x^2 + y^2 + z^2}

    where:
        - :math:`z` is the axial distance from the source to the observation plane,
        - :math:`k = \frac{2\pi}{\lambda}` is the wavenumber.

    Args:
        shape (Vector2): Grid shape (height, width).
        z (Scalar): z-location of the observation plane.
        wavelength (Optional[Scalar]): Wavelength of the wave. If None, uses global default.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the wave. Default: `(0, 0)`.

    Returns:
        torch.Tensor: Real-valued 2D phase argument of the spherical wave.
    """
    wavelength = wavelength_or_default(wavelength)
    z = initialize_tensor("z", z, is_scalar=True)

    x, y = PlanarGrid(shape, spacing=spacing, offset=offset).meshgrid()
    r = torch.sqrt(x**2 + y**2 + z**2)

    k = 2 * torch.pi / wavelength
    return k * r
