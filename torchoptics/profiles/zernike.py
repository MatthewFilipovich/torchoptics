"""This module defines functions to generate Zernike polynomial profiles."""

import math
from typing import Optional

import torch
from torch import Tensor

from ..planar_geometry import PlanarGeometry
from ..type_defs import Int, Scalar, Vector2
from ..utils import initialize_tensor

__all__ = ["zernike"]


def zernike(
    shape: Vector2,
    n: Int,
    m: Int,
    radius: Scalar,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
) -> Tensor:
    r"""
    Generates a Zernike polynomial profile over a 2D grid.

    `Zernike polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_ are a set of orthogonal
    polynomials defined on the unit disk, commonly used in optics for wavefront analysis and aberration
    characterization. They are expressed in polar coordinates :math:`(\rho, \theta)` as:

    .. math::
        Z_n^m(\rho, \theta) = R_n^m(\rho) \exp(i m \theta)

    where:

    - :math:`n` is the radial order (non-negative integer),
    - :math:`m` is the azimuthal index (integer satisfying :math:`|m| \leq n` and :math:`n - |m|` even),
    - :math:`R_n^m(\rho)` is the radial polynomial,
    - :math:`\rho` is the normalized radial coordinate, and
    - :math:`\theta` is the angular coordinate.

    The radial polynomial :math:`R_n^m(\rho)` is computed using:

    .. math::
        R_n^m(\rho) = \sum_{k=0}^{(n-|m|)/2}
        \frac{(-1)^k (n-k)!}{k! \left(\frac{n+|m|}{2}-k\right)! \left(\frac{n-|m|}{2}-k\right)!} \rho^{n-2k}

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        n (int): The radial order of the Zernike polynomial.
        m (int): The azimuthal order of the Zernike polynomial.
        radius (Scalar): The radius of the unit disk.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default.
        offset (Optional[Vector2]): Center coordinates of the profile. Default: `(0, 0)`.

    Returns:
        Tensor: The generated Zernike polynomial profile.
    """

    n = initialize_tensor("n", n, is_scalar=True, is_integer=True, is_non_negative=True)
    m = initialize_tensor("m", m, is_scalar=True, is_integer=True)
    radius = initialize_tensor("radius", radius, is_scalar=True, is_positive=True)

    if m.abs() > n:
        raise ValueError("Azimuthal index m must satisfy |m| <= n.")
    if (n - m.abs()) % 2 != 0:
        raise ValueError("The difference n - |m| must be even for a valid Zernike polynomial.")

    x, y = PlanarGeometry(shape, spacing=spacing, offset=offset).meshgrid()

    rho = torch.sqrt(x**2 + y**2) / radius
    theta = torch.atan2(y, x)

    radial_poly = zernike_radial(n, m.abs(), rho)
    angular_component = torch.cos(m * theta) if m >= 0 else torch.sin(m.abs() * theta)

    # Apply the unit disk mask
    mask = rho <= 1.0
    zernike_profile = radial_poly * angular_component * mask

    return zernike_profile


def zernike_radial(n: Tensor, m: Tensor, rho: Tensor) -> Tensor:
    """Computes the radial component of the Zernike polynomial."""
    radial = torch.zeros_like(rho)
    for k in range((n - m.abs()) // 2 + 1):
        coeff = (-1) ** k * math.comb(n - k, k) * math.comb(n - 2 * k, (n - m.abs()) // 2 - k)
        radial += coeff * rho ** (n - 2 * k)
    return radial
