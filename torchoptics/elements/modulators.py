"""This module defines the modulator elements."""

from typing import Optional

import torch
from torch import Tensor

from ..config import wavelength_or_default
from ..type_defs import Scalar, Vector2
from ..utils import validate_tensor_dim
from .elements import ModulationElement, PolychromaticModulationElement

__all__ = ["Modulator", "PhaseModulator", "AmplitudeModulator", "PolychromaticPhaseModulator"]


class Modulator(ModulationElement):
    """
    Modulator element.

    The modulator is described by a complex modulation profile.

    Args:
        modulation (Tensor): Complex modulation profile.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    modulation: Tensor

    def __init__(
        self,
        modulation: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        validate_tensor_dim(modulation, "modulation", 2)
        super().__init__(modulation.shape, z, spacing, offset)
        self.register_optics_property("modulation", modulation, is_complex=True)

    def modulation_profile(self) -> Tensor:
        return self.modulation


class PhaseModulator(ModulationElement):
    """
    Phase-only modulator element.

    The phase modulator is described by a phase profile.

    Args:
        phase (Tensor): Phase profile (real-valued tensor).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    phase: Tensor

    def __init__(
        self,
        phase: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        validate_tensor_dim(phase, "phase", 2)
        super().__init__(phase.shape, z, spacing, offset)
        self.register_optics_property("phase", phase)

    def modulation_profile(self) -> Tensor:
        return torch.exp(1j * self.phase)


class AmplitudeModulator(ModulationElement):
    """
    Amplitude-only modulator element.

    The amplitude modulator is described by an amplitude profile.

    Args:
        amplitude (Tensor): Amplitude profile (real-valued tensor).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    amplitude: Tensor

    def __init__(
        self,
        amplitude: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        validate_tensor_dim(amplitude, "amplitude", 2)
        super().__init__(amplitude.shape, z, spacing, offset)
        self.register_optics_property("amplitude", amplitude)

    def modulation_profile(self) -> Tensor:
        return self.amplitude.cdouble()  # type: ignore


class PolychromaticPhaseModulator(PolychromaticModulationElement):
    r"""
    Phase-only modulator element that modulates the optical field based on the optical path length (OPL).

    The modulation is applied according to:

    .. math::
        \mathcal{M}(x, y) = \exp\left(i \frac{2 \pi}{\lambda} \cdot \text{OPL}\right)

    where:

    - :math:`\mathcal{M}` is the modulation profile applied to the optical field.
    - :math:`\lambda` is the wavelength of the light.
    - :math:`\text{OPL}` is the optical path length, accounting for both the physical distance and the
      refractive index of the medium.

    Args:
        optical_path_length (Tensor): Optical path length (real-valued tensor).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    optical_path_length: Tensor

    def __init__(
        self,
        optical_path_length: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        validate_tensor_dim(optical_path_length, "optical_path_length", 2)
        super().__init__(optical_path_length.shape, z, spacing, offset)
        self.register_optics_property("optical_path_length", optical_path_length)

    def modulation_profile(self, wavelength: Optional[Scalar] = None) -> Tensor:
        wavelength = wavelength_or_default(wavelength)
        return torch.exp(2j * torch.pi / wavelength * self.optical_path_length)
