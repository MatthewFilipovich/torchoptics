"""This module defines functions to generate spherical and plane wave profiles."""

from typing import Optional

import torch

from ..config import wavelength_or_default
from ..planar_grid import PlanarGrid
from ..type_defs import Scalar, Vector2
from ..utils import initialize_tensor

__all__ = ["plane_wave", "spherical_wave"]


def plane_wave(
    shape: Vector2,
    theta: Scalar = 0.0,
    phi: Scalar = 0.0,
    z: Scalar = 0.0,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
):
    r"""
    Generates a plane wave with arbitrary propagation direction defined by incident angles.

    The wave is defined as:

    .. math::
        \mathcal{M}(x, y) = \exp\left(i (k_x x + k_y y + k_z z) \right)

    where:
        - :math:`k_x = \frac{2\pi}{\lambda} \sin\theta \cos\phi`
        - :math:`k_y = \frac{2\pi}{\lambda} \sin\theta \sin\phi`
        - :math:`k_z = \frac{2\pi}{\lambda} \cos\theta`

    This defines a plane wave propagating in direction \( (\theta, \phi) \) in spherical coordinates.

    Args:
        shape (Vector2): Number of grid points along planar dimensions.
        theta (float): Polar angle from the z-axis, in radians.
        phi (float): Azimuthal angle in the x-y plane, in radians.
        z (Scalar): Axial location at which to evaluate the phase. Default: 0.
        wavelength (Optional[Scalar]): Wavelength of the light. If None, uses global default.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the wave. Default: `(0, 0)`.

    Returns:
        torch.Tensor: Complex-valued 2D phase profile of the plane wave.
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

    phase = kx * x + ky * y + kz * z
    return torch.exp(1j * phase)


def spherical_wave(
    shape: Vector2,
    z: Scalar,
    wavelength: Optional[Scalar] = None,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
    include_amplitude: bool = False,
):
    r"""
    Generates a spherical wave from a point source.

    The field is defined as:

    .. math::
        \mathcal{M}(x, y) = \frac{1}{r} \exp(i k r), \quad
        r = \sqrt{(x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2}

    where:
        - :math:`(x_0, y_0, z_0)` is the source position,
        - :math:`z` is the plane at which the wave is evaluated,
        - :math:`k = \frac{2\pi}{\lambda}` is the wavenumber.

    Args:
        shape (Vector2): Grid shape (height, width).
        z (Scalar): z-location of the observation plane.
        wavelength (Optional[Scalar]): Wavelength of the wave. Defaults to global setting.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the wave. Default: `(0, 0)`.
        include_amplitude (bool): If True, includes 1/r amplitude falloff. Default is False.

    Returns:
        torch.Tensor: Complex-valued spherical wave sampled at the given plane.
    """
    wavelength = wavelength_or_default(wavelength)
    z = initialize_tensor("z", z, is_scalar=True)

    x, y = PlanarGrid(shape, spacing=spacing, offset=offset).meshgrid()
    r = torch.sqrt(x**2 + y**2 + z**2)

    k = 2 * torch.pi / wavelength
    return (1 / r) * torch.exp(1j * k * r) if include_amplitude else torch.exp(1j * k * r)
