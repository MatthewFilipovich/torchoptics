"""This module defines functions to generate grating profiles."""

import math
from typing import Optional

import torch

from ..planar_geometry import PlanarGeometry
from ..type_defs import Vector2

__all__ = ["blazed_grating", "sinusoidal_amplitude_grating", "sinusoidal_phase_grating"]


def blazed_grating(
    shape: Vector2,
    period: float,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
    theta: float = 0,
) -> torch.Tensor:
    r"""
    Generates a blazed grating profile.

    The blazed grating profile is defined by the following equation:

    .. math::
        \mathcal{M}(x, y) = \exp\left( i \frac{2\pi}{\Lambda} (x \cos(\theta) + y \sin(\theta)) \right)

    where:

    - :math:`\Lambda` is the period of the grating, and
    - :math:`\theta` is the angle of the grating.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        period (float): The grating period (distance between adjacent grooves).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.
        theta (float): The angle of the grating in radians. Default: `0`.


    Returns:
        Tensor: The generated transmission function.
    """

    x, y = PlanarGeometry(shape, 0, spacing, offset).meshgrid()
    phase = 2 * torch.pi * (x * math.cos(theta) + y * math.sin(theta)) / period
    return torch.exp(1j * phase)


def sinusoidal_amplitude_grating(
    shape: Vector2,
    m: float,
    period: float,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
    theta: float = 0,
) -> torch.Tensor:
    r"""
    Generates a sinusoidal amplitude grating profile.

    The sinusoidal amplitude grating profile is defined by the following equation:

    .. math::
        \mathcal{M}(x, y) = \frac{1}{2} + \frac{m}{2} \cos\left(2\pi \frac{x \cos(\theta)
        + y \sin(\theta)}{\Lambda}\right)

    where:

    - :math:`m` is the amplitude contrast, and
    - :math:`\Lambda` is the period of the grating.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        m (float): The amplitude contrast.
        period (float): The grating period (distance between adjacent grooves).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.
        theta (float): The angle of the grating in radians. Default: `0`.

    Returns:
        Tensor: The generated transmission function.
    """

    x, y = PlanarGeometry(shape, 0, spacing, offset).meshgrid()
    amplitude = 0.5 + m / 2 * torch.cos(2 * torch.pi * (x * math.cos(theta) + y * math.sin(theta)) / period)
    return amplitude


def sinusoidal_phase_grating(
    shape: Vector2,
    m: float,
    period: float,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
    theta: float = 0,
) -> torch.Tensor:
    r"""
    Generates a sinusoidal phase grating profile.

    The sinusoidal phase grating profile is defined by the following equation:

    .. math::
        \mathcal{M}(x, y) = \exp\left( i \frac{m}{2} \sin\left(2\pi \frac{x \cos(\theta)
        + y \sin(\theta)}{\Lambda}\right) \right)

    where:

    - :math:`m` is the phase contrast, and
    - :math:`\Lambda` is the period of the grating.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        m (float): The phase contrast.
        period (float): The grating period (distance between adjacent grooves).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.
        theta (float): The angle of the grating in radians. Default: `0`.

    Returns:
        Tensor: The generated transmission function.
    """

    x, y = PlanarGeometry(shape, 0, spacing, offset).meshgrid()
    phase = m / 2 * torch.sin(2 * torch.pi * (x * math.cos(theta) + y * math.sin(theta)) / period)
    return torch.exp(1j * phase)
