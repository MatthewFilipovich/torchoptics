"""This module defines a function to generate a Laguerre-Gaussian profile."""

import math
from typing import Callable, Optional

import torch
from torch import Tensor

from ..type_defs import Int, Scalar, Vector2
from ..utils import initialize_tensor
from .hermite_gaussian import calculate_beam_properties

__all__ = ["laguerre_gaussian"]


def laguerre_gaussian(
    shape: Vector2,
    p: Int,
    l: Int,
    waist_radius: Scalar,
    wavelength: Optional[Scalar] = None,
    waist_z: Scalar = 0,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
) -> Tensor:
    r"""
    Generates a Laguerre-Gaussian profile.

    The Laguerre-Gaussian profile is defined by the following equation:

    .. math::
        \begin{align*}
        \psi_{pl}(r, \phi, z) &= \frac{w_0}{w(z)} \left( \frac{\sqrt{2} r}{w(z)} \right)^{|l|} L_p^{|l|}
        \left(\frac{2 r^2}{w(z)^2} \right) \\
        & \quad \times \exp\left(-\frac{r^2}{w(z)^2}\right) \\
        & \quad \times \exp\left(i\left[kz + \frac{kr^2}{2R(z)} - (2p+|l|+1)\arctan\left(\frac{z}{z_R}\right) 
        + l\phi \right]\right)
        \end{align*}

    where the parameters are defined as follows:
    
    - :math:`w_0`: The beam waist radius, representing the minimum beam size at the focal point.
    - :math:`w(z)`: The beam radius at a given propagation distance :math:`z`:

    .. math::

        w(z) = w_0 \sqrt{1 + \left(\frac{z}{z_R}\right)^2}

    - :math:`R(z)`: The radius of curvature of the beam's wavefront at :math:`z`:

    .. math::

        R(z) = z \left(1 + \left(\frac{z_R}{z}\right)^2\right)

    - :math:`z_R`: The Rayleigh range:

    .. math::

        z_R = \frac{\pi w_0^2}{\lambda}

    - :math:`k`: The wavenumber:

    .. math::

        k = \frac{2\pi}{\lambda}

    - :math:`L_p^{|l|}`: The Laguerre polynomial of radial order :math:`p` and azimuthal index :math:`l`.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        p (int): The radial mode number.
        l (int): The azimuthal mode number.
        waist_radius: The radius of the beam waist.
        wavelength (Scalar, optional): The wavelength of the beam. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        waist_z: Position of beam waist along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.

    Returns:
        Tensor: The generated Laguerre-Gaussian profile.
    """
    # pylint: disable=too-many-locals
    p = initialize_tensor("p", p, is_scalar=True, is_integer=True, is_non_negative=True)
    l = initialize_tensor("l", l, is_scalar=True, is_integer=True)
    waist_radius = initialize_tensor("waist_radius", waist_radius, is_scalar=True, is_positive=True)

    x, y, phase_shift, wz, waist_ratio = calculate_beam_properties(
        (p, l), waist_radius, shape, wavelength, waist_z, spacing, offset, False
    )

    r = torch.sqrt(x**2 + y**2) * 2.0**0.5 / wz
    phi = torch.atan2(y, x)
    normalization_constant = torch.sqrt(
        2 * math.factorial(p) / (torch.pi * math.factorial(p + abs(l))) / waist_radius**2
    )

    return (
        waist_ratio
        * (r ** abs(l))
        * torch.exp(-(r**2) / 2)
        * torch.exp(1j * l * phi)
        * laguerre_poly(p, abs(l))(r**2)
        * torch.exp(1j * phase_shift)
        * normalization_constant
    )


def laguerre_poly(p: Int, l: Int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compute the generalized Laguerre polynomial L_p^l(x)."""

    def poly(x):
        if p == 0:
            return torch.ones_like(x)
        if p == 1:
            return 1 + l - x
        laguerre_poly_p_minus_1 = laguerre_poly(p - 1, l)
        laguerre_poly_p_minus_2 = laguerre_poly(p - 2, l)
        return (
            (2 * p - 1 + l - x) * laguerre_poly_p_minus_1(x) - (p - 1 + l) * laguerre_poly_p_minus_2(x)
        ) / p

    return poly
