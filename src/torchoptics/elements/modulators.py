"""Modulator element definitions."""

from collections.abc import Callable

import torch
from torch import Tensor

from ..types import Scalar, Vector2
from ..utils import validate_tensor_ndim, wavelength_or_default
from .elements import ModulationElement, PolychromaticModulationElement


class Modulator(ModulationElement):
    """Modulator element.

    The modulator is described by a complex modulation profile.

    Args:
        modulation (Tensor): Complex modulation profile.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the plane. Default: `(0, 0)`.

    """

    modulation: Tensor

    def __init__(
        self,
        modulation: Tensor,
        z: Scalar = 0,
        spacing: Vector2 | None = None,
        offset: Vector2 | None = None,
    ) -> None:
        """Initialize the Modulator."""
        validate_tensor_ndim(modulation, "modulation", 2)
        super().__init__(modulation.shape, z, spacing, offset)
        self.register_optics_property("modulation", modulation, is_complex=True)

    def modulation_profile(self) -> Tensor:
        """Return the modulation profile."""
        return self.modulation


class PhaseModulator(ModulationElement):
    """Phase-only modulator element.

    The phase modulator is described by a phase profile.

    Args:
        phase (Tensor): Phase profile (real-valued tensor).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the plane. Default: `(0, 0)`.

    """

    phase: Tensor

    def __init__(
        self,
        phase: Tensor,
        z: Scalar = 0,
        spacing: Vector2 | None = None,
        offset: Vector2 | None = None,
    ) -> None:
        """Initialize the PhaseModulator."""
        validate_tensor_ndim(phase, "phase", 2)
        super().__init__(phase.shape, z, spacing, offset)
        self.register_optics_property("phase", phase)

    def modulation_profile(self) -> Tensor:
        """Return the modulation profile."""
        return torch.exp(1j * self.phase)


class AmplitudeModulator(ModulationElement):
    """Amplitude-only modulator element.

    The amplitude modulator is described by an amplitude profile.

    Args:
        amplitude (Tensor): Amplitude profile (real-valued tensor).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the plane. Default: `(0, 0)`.

    """

    amplitude: Tensor

    def __init__(
        self,
        amplitude: Tensor,
        z: Scalar = 0,
        spacing: Vector2 | None = None,
        offset: Vector2 | None = None,
    ) -> None:
        """Initialize the AmplitudeModulator."""
        validate_tensor_ndim(amplitude, "amplitude", 2)
        super().__init__(amplitude.shape, z, spacing, offset)
        self.register_optics_property("amplitude", amplitude)

    def modulation_profile(self) -> Tensor:
        """Return the modulation profile."""
        return self.amplitude + 0j


class PolychromaticPhaseModulator(PolychromaticModulationElement):
    r"""Phase-only modulator element that modulates the optical field based on physical thickness.

    The modulation is applied according to:

    .. math::
        \mathcal{M}(x, y) = \exp\left(i \frac{2 \pi}{\lambda}
        \left[ n\left(\lambda\right) - 1 \right] t(x, y)\right)

    where:

    - :math:`\mathcal{M}` is the modulation profile applied to the optical field.
    - :math:`\lambda` is the wavelength of the light.
    - :math:`n(\lambda)` is the wavelength-dependent refractive index of the medium.
    - :math:`t(x, y)` is the physical thickness of the medium at each point.

    Args:
        thickness (Tensor): Physical thickness of the medium (real-valued tensor).
        n (Scalar | Callable[[Scalar], Scalar]): Refractive index. Can be a scalar (constant) or a
            callable that takes the wavelength and returns the refractive index (for dispersive media).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the plane. Default: `(0, 0)`.

    """

    thickness: Tensor
    n: Callable[[Scalar], Scalar]

    def __init__(
        self,
        thickness: Tensor,
        n: Scalar | Callable[[Scalar], Scalar],
        z: Scalar = 0,
        spacing: Vector2 | None = None,
        offset: Vector2 | None = None,
    ) -> None:
        """Initialize the PolychromaticPhaseModulator."""
        validate_tensor_ndim(thickness, "thickness", 2)
        super().__init__(thickness.shape, z, spacing, offset)
        self.register_optics_property("thickness", thickness)
        self.n = n if isinstance(n, Callable) else lambda _: n

    def modulation_profile(self, wavelength: Scalar | None = None) -> Tensor:
        """Return the modulation profile."""
        wavelength = wavelength_or_default(wavelength)
        return torch.exp(2j * torch.pi / wavelength * (self.n(wavelength) - 1) * self.thickness)
