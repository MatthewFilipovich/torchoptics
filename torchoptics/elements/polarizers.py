"""This module defines the polarizer elements."""

from typing import Optional

import torch
from torch import Tensor

from ..type_defs import Scalar, Vector2
from .elements import PolarizedModulationElement

__all__ = ["LinearPolarizer", "LeftCircularPolarizer", "RightCircularPolarizer"]


class LinearPolarizer(PolarizedModulationElement):
    r"""
    Linear polarizer element.

    The linear polarizer is described by the following polarization matrix:

    .. math::
        J = 
        \begin{bmatrix}
            \cos^2(\theta) & \cos(\theta) \sin(\theta) & 0 \\
            \cos(\theta) \sin(\theta) & \sin^2(\theta) & 0 \\
            0 & 0 & 1
        \end{bmatrix}

    where :math:`\theta` is the transmission axis of the polarizer.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        theta (Scalar): Transmission axis of the polarizer.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    theta: Tensor

    def __init__(
        self,
        shape: Vector2,
        theta: Scalar,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        super().__init__(shape, z, spacing, offset)
        self.register_optics_property("theta", theta, is_scalar=True)

    def polarized_modulation_profile(self) -> Tensor:
        tensor = torch.zeros(3, 3, *self.shape, dtype=torch.cdouble, device=next(self.buffers()).device)
        tensor[0, 0] = torch.cos(self.theta) ** 2
        tensor[0, 1] = torch.cos(self.theta) * torch.sin(self.theta)
        tensor[1, 0] = torch.cos(self.theta) * torch.sin(self.theta)
        tensor[1, 1] = torch.sin(self.theta) ** 2
        tensor[2, 2] = 1
        return tensor


class LeftCircularPolarizer(PolarizedModulationElement):
    r"""
    Left circular polarizer element.

    The left circular polarizer is described by the following polarization matrix:

    .. math::
        J = 
        \begin{bmatrix}
            \frac{1}{2} & -\frac{i}{2} & 0 \\
            \frac{1}{2} & \frac{1}{2} & 0 \\
            0 & 0 & 1
        \end{bmatrix}

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def polarized_modulation_profile(self) -> Tensor:
        tensor = torch.zeros(3, 3, *self.shape, dtype=torch.cdouble, device=next(self.buffers()).device)
        tensor[0, 0] = 0.5
        tensor[0, 1] = -0.5j  # type: ignore[assignment]
        tensor[1, 0] = 0.5j  # type: ignore[assignment]
        tensor[1, 1] = 0.5
        tensor[2, 2] = 1
        return tensor


class RightCircularPolarizer(PolarizedModulationElement):
    r"""
    Right circular polarizer element.

    The right circular polarizer is described by the following polarization matrix:

    .. math::
        J =
        \begin{bmatrix}
            \frac{1}{2} & \frac{i}{2} & 0 \\
            \frac{-i}{2} & \frac{1}{2} & 0 \\
            0 & 0 & 1
        \end{bmatrix}
    
    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def polarized_modulation_profile(self) -> Tensor:
        tensor = torch.zeros(3, 3, *self.shape, dtype=torch.cdouble, device=next(self.buffers()).device)
        tensor[0, 0] = 0.5
        tensor[0, 1] = 0.5j  # type: ignore[assignment]
        tensor[1, 0] = -0.5j  # type: ignore[assignment]
        tensor[1, 1] = 0.5
        tensor[2, 2] = 1
        return tensor
