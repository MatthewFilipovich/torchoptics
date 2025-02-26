"""This module defines functions to generate Hermite-Gaussian and Gaussian profiles."""

import math
from typing import Callable, Optional

import torch
from torch import Tensor

from ..config import wavelength_or_default
from ..planar_geometry import PlanarGeometry
from ..type_defs import Int, Scalar, Vector2
from ..utils import initialize_tensor

__all__ = ["hermite_gaussian", "gaussian"]


def hermite_gaussian(
    shape: Vector2,
    m: Int,
    n: Int,
    waist_radius: Scalar,
    wavelength: Optional[Scalar] = None,
    waist_z: Scalar = 0,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
) -> Tensor:
    r"""
    Generates a Hermite-Gaussian profile.

    The Hermite-Gaussian profile is defined by the following equation:

    .. math::
        \begin{align*}
        \psi_{mn}(x, y, z) &= \frac{w_0}{w(z)} H_m\left(\sqrt{2}\frac{x}{w(z)}\right) 
        H_n\left(\sqrt{2}\frac{y}{w(z)}\right) \\
        & \quad \times \exp\left(-\frac{x^2 + y^2}{w(z)^2}\right) \\
        & \quad \times \exp\left(i\left[kz + \frac{k(x^2 + y^2)}{2R(z)} 
        - (m+n+1)\arctan\left(\frac{z}{z_R}\right)\right]\right),
        \end{align*}

    where the parameters are defined as follows:

    - :math:`w_0`: The beam waist radius, representing the minimum beam size at the focal point.
    - :math:`w(z)`: The beam radius at a given propagation distance :math:`z`:

    .. math::

        w(z) = w_0 \sqrt{1 + \left(\frac{z}{z_R}\right)^2}

    - :math:`R(z)`: The radius of curvature of the beam's wavefront at a distance :math:`z`:

    .. math::

        R(z) = z \left(1 + \left(\frac{z_R}{z}\right)^2\right)

    - :math:`z_R`: The Rayleigh range:

    .. math::

        z_R = \frac{\pi w_0^2}{\lambda}

    - :math:`k`: The wavenumber:

    .. math::

        k = \frac{2\pi}{\lambda}

    - :math:`H_m` and :math:`H_n`: The Hermite polynomials of order :math:`m` and :math:`n`.
    
    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        m (int): The mode number in the first planar dimension.
        n (int): The mode number in the second planar dimension.
        waist_radius: The radius of the beam waist.
        wavelength (Scalar, optional): The wavelength of the beam. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        waist_z: Position of beam waist along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.

    Returns:
        Tensor: The generated Hermite-Gaussian profile.
    """
    # pylint: disable=too-many-locals

    m = initialize_tensor("m", m, is_scalar=True, is_integer=True, is_non_negative=True)
    n = initialize_tensor("n", n, is_scalar=True, is_integer=True, is_non_negative=True)
    waist_radius = initialize_tensor("waist_radius", waist_radius, is_scalar=True, is_positive=True)

    x, y, phase_shift, wz, waist_ratio = calculate_beam_properties(
        (m, n), waist_radius, shape, wavelength, waist_z, spacing, offset, True
    )

    u_x = 2.0**0.5 * x / wz
    u_y = 2.0**0.5 * y / wz
    normalization_constant = torch.sqrt(
        2.0 ** (1 - m - n) / (math.factorial(m) * math.factorial(n) * torch.pi * waist_radius**2)
    )

    return (
        waist_ratio
        * hermite_poly(m)(u_x)
        * hermite_poly(n)(u_y)
        * torch.exp(-(u_x**2 + u_y**2) / 2)
        * torch.exp(1j * phase_shift)
        * normalization_constant
    )


def gaussian(
    shape: Vector2,
    waist_radius: Scalar,
    wavelength: Optional[Scalar] = None,
    waist_z: Scalar = 0,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
) -> Tensor:
    r"""
    Generates a Gaussian profile.

    The Gaussian profile is defined by the following equation:

    .. math::
        \psi(r, z) = \frac{w_0}{w(z)} \exp\left(-\frac{r^2}{w(z)^2}\right)
        \exp\left(i\left[kz + \frac{kr^2}{2R(z)} - \arctan\left(\frac{z}{z_R}\right)\right]\right)

    where the parameters are defined as follows:

    - :math:`w_0`: The beam waist radius, representing the minimum beam size at the focal point.
    - :math:`w(z)`: The beam radius at a given propagation distance :math:`z`:

    .. math::

        w(z) = w_0 \sqrt{1 + \left(\frac{z}{z_R}\right)^2}

    - :math:`R(z)`: The radius of curvature of the beam's wavefront at a distance :math:`z`:

    .. math::

        R(z) = z \left(1 + \left(\frac{z_R}{z}\right)^2\right)

    - :math:`z_R`: The Rayleigh range:

    .. math::

        z_R = \frac{\pi w_0^2}{\lambda}

    - :math:`k`: The wavenumber:

    .. math::

        k = \frac{2\pi}{\lambda}

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        waist_radius: The radius of the beam waist.
        wavelength (Scalar, optional): The wavelength of the beam. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        waist_z: Position of beam waist along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.

    Returns:
        Tensor: The generated Gaussian profile.

    """
    return hermite_gaussian(shape, 0, 0, waist_radius, wavelength, waist_z, spacing, offset)


def hermite_poly(n: Int) -> Callable[[Tensor], Tensor]:
    """Compute the Hermite polynomial of order n using an iterative approach."""

    def poly(x: Tensor) -> Tensor:
        h0 = torch.ones_like(x)
        if n == 0:
            return h0
        h1 = 2 * x
        if n == 1:
            return h1
        for k in range(2, n + 1):
            hk = 2 * x * h1 - 2 * (k - 1) * h0
            h0, h1 = h1, hk
        return h1

    return poly


def calculate_beam_properties(mode_nums, waist_radius, shape, wavelength, waist_z, spacing, offset, is_hg):
    """Calculate the properties of the beam."""
    wavelength = get_wavelength(wavelength, waist_z)
    x, y, z = calculate_coordinates(shape, waist_z, spacing, offset)
    z_div_rayleigh_range = calculate_z_div_rayleigh_range(z, waist_radius, wavelength)
    phase_shift = calculate_phase_shift(mode_nums, wavelength, x, y, z, z_div_rayleigh_range, is_hg)
    wz = calculate_wz(waist_radius, z_div_rayleigh_range)
    waist_ratio = calculate_waist_ratio(z_div_rayleigh_range)
    return x, y, phase_shift, wz, waist_ratio


def get_wavelength(wavelength, waist_z):
    """Get the wavelength of the beam."""
    if waist_z == 0:  # Wavelength does not matter at the waist (can be None)
        return wavelength
    return wavelength_or_default(wavelength)


def calculate_coordinates(shape, waist_z, spacing, offset):
    """Calculate the coordinates of the beam."""
    planar_geometry = PlanarGeometry(shape, waist_z, spacing, offset)
    x, y = planar_geometry.meshgrid()
    z = planar_geometry.z
    return x, y, z


def calculate_z_div_rayleigh_range(z, waist_radius, wavelength):
    """Calculate z divided by the Rayleigh range."""
    if z == 0:
        return 0
    return z / (torch.pi * waist_radius**2 / wavelength)


def calculate_phase_shift(mode_nums, wavelength, x, y, z, z_div_rayleigh_range, is_hg):
    """Calculate the phase shift of the beam."""
    if z == 0:
        return torch.tensor(0, dtype=torch.double)
    radius_of_curvature = z * (1 + z_div_rayleigh_range ** (-2))
    wave_number = 2.0 * torch.pi / wavelength
    radial_distance = torch.sqrt(x**2 + y**2)
    phase_shift = wave_number * z + wave_number * radial_distance**2 / (2 * radius_of_curvature)
    if is_hg:
        phase_shift -= (mode_nums[0] + mode_nums[1] + 1) * torch.atan(z_div_rayleigh_range)
    else:  # laguerre-gaussian
        phase_shift -= (2 * mode_nums[0] + abs(mode_nums[1]) + 1) * torch.atan(z_div_rayleigh_range)

    return phase_shift


def calculate_wz(waist_radius, z_div_rayleigh_range):
    """Calculate the beam waist radius."""
    return waist_radius * (1 + z_div_rayleigh_range**2) ** 0.5


def calculate_waist_ratio(z_div_rayleigh_range):
    """Calculate the ratio of the beam waist radius to the beam waist radius at the waist."""
    return 1 / (1 + z_div_rayleigh_range**2) ** 0.5
