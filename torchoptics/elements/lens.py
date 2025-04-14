"""This module defines the Lens element."""

from typing import Optional

import torch
from torch import Tensor

from ..profiles import circle, cylindrical_lens_phase, lens_phase
from ..type_defs import Scalar, Vector2
from .elements import PolychromaticModulationElement

__all__ = ["Lens", "CylindricalLens"]


class Lens(PolychromaticModulationElement):
    r"""
    Lens element.

    Represents a thin lens with the following modulation profile:

    .. math::
        \mathcal{M}(x, y) = \exp\left(-i \frac{\pi}{\lambda f} (x^2 + y^2) \right)

    where:
        - :math:`\lambda` is the wavelength of the light, and
        - :math:`f` is the focal length of the lens.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        focal_length (Scalar): Focal length of the lens.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    focal_length: Tensor

    def __init__(
        self,
        shape: Vector2,
        focal_length: Scalar,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        super().__init__(shape, z, spacing, offset)
        self.register_optics_property("focal_length", focal_length, is_scalar=True)

    def modulation_profile(self, wavelength: Optional[Scalar] = None) -> Tensor:
        phase = lens_phase(self.shape, self.focal_length, wavelength, self.spacing)
        radius = self.length().min() / 2
        amplitude = circle(self.shape, radius, self.spacing)
        return amplitude * torch.exp(1j * phase)


class CylindricalLens(PolychromaticModulationElement):
    r"""
    Cylindrical lens element.

    Represents a thin cylindrical lens with the following modulation profile:

    .. math::
        \mathcal{M}(x, y) = \exp\left(-i \frac{\pi}{\lambda f} x^2 \right)

    where:
        - :math:`\lambda` is the wavelength of the light, and
        - :math:`f` is the focal length of the lens.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        focal_length (Scalar): Focal length of the lens.
        z (Scalar): Position along the z-axis. Default: `0`.
        theta (Scalar): Angle of the lens in radians. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    focal_length: Tensor
    theta: Tensor

    def __init__(
        self,
        shape: Vector2,
        focal_length: Scalar,
        theta: Scalar = 0,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        super().__init__(shape, z, spacing, offset)
        self.register_optics_property("focal_length", focal_length, is_scalar=True)
        self.register_optics_property("theta", theta, is_scalar=True)

    def modulation_profile(self, wavelength: Optional[Scalar] = None) -> Tensor:
        phase = cylindrical_lens_phase(self.shape, self.focal_length, self.theta, wavelength, self.spacing)
        radius = self.length().min() / 2
        amplitude = circle(self.shape, radius, self.spacing)
        return amplitude * torch.exp(1j * phase)
