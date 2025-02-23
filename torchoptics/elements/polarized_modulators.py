"""This module defines the polarized modulator elements."""

from typing import Optional

import torch
from torch import Tensor

from ..type_defs import Scalar, Vector2
from ..utils import validate_tensor_dim
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

    polarized_modulation: Tensor

    def __init__(
        self,
        polarized_modulation: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        _validate_tensor(polarized_modulation, "polarized_modulation")
        super().__init__(polarized_modulation.shape[2:], z, spacing, offset)
        self.register_optics_property("polarized_modulation", polarized_modulation, is_complex=True)

    def polarized_modulation_profile(self) -> Tensor:
        return self.polarized_modulation


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

    phase: Tensor

    def __init__(
        self,
        phase: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        _validate_tensor(phase, "phase")
        super().__init__(phase.shape[2:], z, spacing, offset)
        self.register_optics_property("phase", phase)

    def polarized_modulation_profile(self) -> Tensor:
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

    amplitude: Tensor

    def __init__(
        self,
        amplitude: Tensor,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        _validate_tensor(amplitude, "amplitude")
        super().__init__(amplitude.shape[2:], z, spacing, offset)
        self.register_optics_property("amplitude", amplitude)

    def polarized_modulation_profile(self) -> Tensor:
        return self.amplitude.cdouble()  # type: ignore


def _validate_tensor(tensor, name):
    validate_tensor_dim(tensor, name, 4)
    if tensor.shape[:2] != (3, 3):
        raise ValueError(
            f"Expected first two dimensions of {name} to have shape (3, 3), but got {tensor.shape[:2]}"
        )
