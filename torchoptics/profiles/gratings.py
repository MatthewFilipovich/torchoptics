"""This module defines functions to generate grating profiles."""

from typing import Optional

import torch

from ..planar_geometry import PlanarGeometry
from ..type_defs import Scalar, Vector2
from ..utils import initialize_tensor

__all__ = ["binary_grating", "blazed_grating", "sinusoidal_grating"]


def binary_grating(
    shape: Vector2,
    period: Scalar,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
    height: Scalar = 1,
    theta: Scalar = 0,
    duty_cycle: Scalar = 0.5,
) -> torch.Tensor:
    r"""
    Generates a binary grating profile.

    The binary grating profile is defined by the following equation:

    .. math::
        \psi(x, y) = 
        \begin{cases} 
        0 & \text{if } \mod \left(\frac{x \cos \theta + y \sin \theta}{\Lambda}, 1\right) < d \\
        h & \text{otherwise} 
        \end{cases}

    where:

    - :math:`h` is the height of the grating,
    - :math:`\Lambda` is the period of the grating,
    - :math:`\theta` is the angle of the grating, and
    - :math:`d` is the duty cycle of the grating.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        period (Scalar): The grating period (distance between adjacent grooves).
        duty_cycle (Scalar): The duty cycle of the grating. Default: `0.5`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.
        height (Scalar): The height of the grating. Default: `1`.
        theta (Scalar): The angle of the grating in radians. Default: `0`.

    Returns:
        Tensor: The generated transmission function.
    """

    duty_cycle = initialize_tensor("duty_cycle", duty_cycle, is_scalar=True)
    height = initialize_tensor("height", height, is_scalar=True)
    theta = initialize_tensor("theta", theta, is_scalar=True)

    grating = blazed_grating(shape, period, spacing, offset, 1, theta)
    return torch.where(grating < duty_cycle, 0.0, height)


def blazed_grating(
    shape: Vector2,
    period: Scalar,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
    height: Scalar = 1,
    theta: Scalar = 0,
) -> torch.Tensor:
    r"""
    Generates a blazed grating profile.

    The blazed grating profile is defined by the following equation:

    .. math::
        \psi(x, y) = h \cdot \mod \left(\frac{x \cos \theta + y \sin \theta}{\Lambda}, 1\right)

    where:

    - :math:`h` is the height of the grating,
    - :math:`\Lambda` is the period of the grating, and
    - :math:`\theta` is the angle of the grating.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        period (Scalar): The grating period (distance between adjacent grooves).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.
        height (Scalar): The height of the grating. Default: `1`.
        theta (Scalar): The angle of the grating in radians. Default: `0`.

    Returns:
        Tensor: The generated transmission function.
    """

    period = initialize_tensor("period", period, is_scalar=True)
    height = initialize_tensor("height", height, is_scalar=True)
    theta = initialize_tensor("theta", theta, is_scalar=True)
    x, y = PlanarGeometry(shape, spacing=spacing, offset=offset).meshgrid()

    grating = ((x * torch.cos(theta) + y * torch.sin(theta)) / period) % 1
    grating = grating.where(grating < 1 - 1e-10, 0.0)  # Avoid numerical issues from modulus
    return height * grating


def sinusoidal_grating(
    shape: Vector2,
    period: Scalar,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
    height: Scalar = 1,
    theta: Scalar = 0,
) -> torch.Tensor:
    r"""
    Generates a sinusoidal grating profile.

    The sinusoidal grating profile is defined by the following equation:

    .. math::
        \psi(x, y) = h \left(\frac{1}{2} + \frac{1}{2} \cos\left(\frac{2 \pi}{\Lambda} (x \cos \theta
        + y \sin \theta)\right)\right)

    where:

    - :math:`h` is the height of the grating,
    - :math:`\Lambda` is the period of the grating, and
    - :math:`\theta` is the angle of the grating.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        period (Scalar): The grating period (distance between adjacent grooves).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the beam. Default: `(0, 0)`.
        height (Scalar): The height of the grating. Default: `1`.
        theta (Scalar): The angle of the grating in radians. Default: `0`.

    Returns:
        Tensor: The generated transmission function.
    """

    period = initialize_tensor("period", period, is_scalar=True)
    height = initialize_tensor("height", height, is_scalar=True)
    theta = initialize_tensor("theta", theta, is_scalar=True)
    x, y = PlanarGeometry(shape, spacing=spacing, offset=offset).meshgrid()

    grating = 0.5 + 0.5 * torch.cos(2 * torch.pi * (x * torch.cos(theta) + y * torch.sin(theta)) / period)
    return height * grating
