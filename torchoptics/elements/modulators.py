"""This module defines the modulator elements."""

from typing import Optional

import torch
from torch import Tensor

from ..param import Param
from ..type_defs import Scalar, Vector2
from .elements import ModulationElement

__all__ = ["Modulator", "PhaseModulator", "AmplitudeModulator"]


class Modulator(ModulationElement):
    """
    Modulator element.

    The modulator is described by a complex modulation profile.

    Args:
        modulation_profile (Tensor): Complex modulation profile.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def __init__(
        self,
        modulation_profile: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        _validate_input_tensor("modulation_profile", modulation_profile)
        super().__init__(modulation_profile.shape, z, spacing, offset)
        self.register_optics_property("modulation_profile", modulation_profile, is_complex=True)


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

    def __init__(
        self,
        phase: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        _validate_input_tensor("phase", phase)
        super().__init__(phase.shape, z, spacing, offset)
        self.register_optics_property("phase", phase)

    @property
    def modulation_profile(self) -> Tensor:
        """Returns the phase modulation profile."""
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

    def __init__(
        self,
        amplitude: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        _validate_input_tensor("amplitude", amplitude)
        super().__init__(amplitude.shape, z, spacing, offset)
        self.register_optics_property("amplitude", amplitude)

    @property
    def modulation_profile(self) -> Tensor:
        """Returns the amplitude modulation profile."""
        return self.amplitude.cdouble()


def _validate_input_tensor(name, tensor):
    if not isinstance(tensor, (Tensor, Param)):
        raise TypeError(f"Expected {name} to be a tensor, but got {type(tensor).__name__}")
    if tensor.dim() != 2:
        raise ValueError(f"Expected {name} to be a 2D tensor, but got {tensor.dim()}D")
