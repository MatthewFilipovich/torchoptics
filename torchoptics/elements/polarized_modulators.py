"""This module defines the polarized modulator elements."""

from typing import Optional

import torch
from torch import Tensor

from ..param import Param
from ..type_defs import Scalar, Vector2
from .elements import PolarizedModulationElement

__all__ = ["PolarizedModulator", "PolarizedPhaseModulator", "PolarizedAmplitudeModulator"]


class PolarizedModulator(PolarizedModulationElement):
    """
    Polarized modulator element.

    The polarized modulator is described by a complex polarized modulation profile.

    Args:
        polarized_modulation_profile (Tensor): Complex 3x3 polarized modulation profile.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def __init__(
        self,
        polarized_modulation_profile: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        _validate_input_tensor(polarized_modulation_profile, "polarized_modulation_profile")
        super().__init__(polarized_modulation_profile.shape[2:], z, spacing, offset)
        self.register_optics_property(
            "polarized_modulation_profile", polarized_modulation_profile, is_complex=True
        )


class PolarizedPhaseModulator(PolarizedModulationElement):
    """
    Polarized phase-only modulator element.

    The polarized phase modulator is described by a polarized phase profile.

    Args:
        phase (Tensor): Phase profile (real-valued 3x3 tensor).
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
        _validate_input_tensor(phase, "phase")
        super().__init__(phase.shape[2:], z, spacing, offset)
        self.register_optics_property("phase", phase)

    @property
    def polarized_modulation_profile(self) -> Tensor:
        """Returns the polarized modulation profile as a complex tensor."""
        return torch.exp(1j * self.phase)


class PolarizedAmplitudeModulator(PolarizedModulationElement):
    """
    Polarized amplitude-only modulator element.

    The polarized amplitude modulator is described by an amplitude profile.

    Args:
        amplitude (Tensor): Amplitude profile (real-valued 3x3 tensor).
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
        _validate_input_tensor(amplitude, "amplitude")
        super().__init__(amplitude.shape[2:], z, spacing, offset)
        self.register_optics_property("amplitude", amplitude)

    @property
    def polarized_modulation_profile(self) -> Tensor:
        """Returns the polarized modulation profile as a complex tensor."""
        return self.amplitude.cdouble()


def _validate_input_tensor(tensor, name):
    if not isinstance(tensor, (Tensor, Param)):
        raise TypeError(f"Expected {name} to be a tensor, but got {type(tensor).__name__}")
    if tensor.dim() != 4:
        raise ValueError(f"Expected {name} to be a 4D tensor, but got {tensor.dim()}D")
    if tensor.shape[:2] != (3, 3):
        raise ValueError(
            f"Expected first two dimensions of {name} to have shape (3, 3), but got {tensor.shape[:2]}"
        )
