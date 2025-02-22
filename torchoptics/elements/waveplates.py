"""This module defines the waveplate element."""

from typing import Optional

import torch
from torch import Tensor

from ..type_defs import Scalar, Vector2
from .elements import PolarizedModulationElement

__all__ = ["Waveplate", "QuarterWaveplate", "HalfWaveplate"]


class Waveplate(PolarizedModulationElement):
    r"""
    Waveplate element.

    The waveplate is described by the following polarization matrix:

    .. math::
        J = 
        \begin{bmatrix}
            \cos^2(\theta) + e^{i \phi} \sin^2(\theta) & (1 - e^{i \phi}) \cos(\theta) \sin(\theta) & 0 \\
            (1 - e^{i \phi}) \cos(\theta) \sin(\theta) & \sin^2(\theta) + e^{i \phi} \cos^2(\theta) & 0 \\
            0 & 0 & 1
        \end{bmatrix}

    where 
        - :math:`\theta` is the fast axis angle, and 
        - :math:`\phi` is the phase delay between the fast and slow axes.

    .. note::
        A quarter waveplate (QWP) is obtained by setting :math:`\phi = \pi/2`. 

        A half waveplate (HWP) is obtained by setting :math:`\phi = \pi`. 

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        phi (Scalar): Phase delay of the waveplate.
        theta (Scalar): Fast axis angle of the waveplate.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    phi: Tensor
    theta: Tensor

    def __init__(
        self,
        shape: Vector2,
        phi: Scalar,
        theta: Scalar,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        super().__init__(shape, z, spacing, offset)
        self.register_optics_property("phi", phi, is_scalar=True)
        self.register_optics_property("theta", theta, is_scalar=True)

    def polarized_modulation_profile(self) -> Tensor:
        tensor = torch.zeros(3, 3, *self.shape, dtype=torch.cdouble, device=next(self.buffers()).device)
        tensor[0, 0] = torch.cos(self.theta) ** 2 + torch.exp(1j * self.phi) * torch.sin(self.theta) ** 2
        tensor[0, 1] = (1 - torch.exp(1j * self.phi)) * torch.cos(self.theta) * torch.sin(self.theta)
        tensor[1, 0] = (1 - torch.exp(1j * self.phi)) * torch.cos(self.theta) * torch.sin(self.theta)
        tensor[1, 1] = torch.sin(self.theta) ** 2 + torch.exp(1j * self.phi) * torch.cos(self.theta) ** 2
        tensor[2, 2] = 1
        return tensor


class QuarterWaveplate(PolarizedModulationElement):
    r"""
    Quarter Waveplate element.

    The quarter waveplate is described by the following polarization matrix:

    .. math::
        J = 
        \begin{bmatrix}
            \cos^2(\theta) + i \sin^2(\theta) & (1 - i) \cos(\theta) \sin(\theta) & 0 \\
            (1 - i) \cos(\theta) \sin(\theta) & \sin^2(\theta) + i \cos^2(\theta) & 0 \\
            0 & 0 & 1
        \end{bmatrix}

    where 
        - :math:`\theta` is the fast axis angle.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        theta (Scalar): Fast axis angle of the waveplate.
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
        tensor[0, 0] = torch.cos(self.theta) ** 2 + 1j * torch.sin(self.theta) ** 2
        tensor[0, 1] = (1 - 1j) * torch.cos(self.theta) * torch.sin(self.theta)
        tensor[1, 0] = (1 - 1j) * torch.cos(self.theta) * torch.sin(self.theta)
        tensor[1, 1] = torch.sin(self.theta) ** 2 + 1j * torch.cos(self.theta) ** 2
        tensor[2, 2] = 1
        return tensor


class HalfWaveplate(PolarizedModulationElement):
    r"""
    Half Waveplate element.

    The half waveplate is described by the following polarization matrix:

    .. math::
        J = 
        \begin{bmatrix}
            \cos^2(\theta) - \sin^2(\theta) & 2 \cos(\theta) \sin(\theta) & 0 \\
            2 \cos(\theta) \sin(\theta) & \sin^2(\theta) - \cos^2(\theta) & 0 \\
            0 & 0 & 1
        \end{bmatrix}

    where 
        - :math:`\theta` is the fast axis angle.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        theta (Scalar): Fast axis angle of the waveplate.
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
        tensor[0, 0] = torch.cos(self.theta) ** 2 - torch.sin(self.theta) ** 2
        tensor[0, 1] = 2 * torch.cos(self.theta) * torch.sin(self.theta)
        tensor[1, 0] = 2 * torch.cos(self.theta) * torch.sin(self.theta)
        tensor[1, 1] = torch.sin(self.theta) ** 2 - torch.cos(self.theta) ** 2
        tensor[2, 2] = 1
        return tensor
