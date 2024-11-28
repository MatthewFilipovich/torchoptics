"""This module defines the Lens element."""

from typing import Optional

from torch import Tensor

from ..config import get_default_wavelength
from ..profiles import lens
from ..type_defs import Scalar, Vector2
from .elements import ModulationElement

__all__ = ["Lens"]


class Lens(ModulationElement):
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
        z (Scalar): Position along the z-axis. Default: `0`.
        focal_length (Scalar): Focal length of the lens.
        wavelength (Optional[Scalar]): Wavelength used for lens operation. Default: if `None`, uses a
        global default (see :meth:`torchoptics.config.set_default_wavelength()`).
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
        is_circular_lens (bool): If `True`, the lens is circular and the phase profile is set to zero outside
            the lens diameter, otherwise lens is square. Default: `True`.
    """

    def __init__(
        self,
        shape: Vector2,
        focal_length: Scalar,
        z: Scalar = 0,
        wavelength: Optional[Scalar] = None,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
        is_circular_lens: bool = True,
    ) -> None:
        super().__init__(shape, z, spacing, offset)
        self.register_optics_property("focal_length", focal_length, ())
        self.register_optics_property(
            "wavelength",
            get_default_wavelength() if wavelength is None else wavelength,
            (),
            validate_positive=True,
        )
        self.is_circular_lens = is_circular_lens

    @property
    def modulation_profile(self) -> Tensor:
        """Returns the phase modulation profile of the lens."""
        return lens(
            self.shape,
            self.focal_length,
            self.z,
            self.wavelength,
            self.spacing,
            None,  # Offset is not used in lens profile
            self.is_circular_lens,
        )

    @property
    def is_circular_lens(self) -> bool:
        """Returns whether the lens is circular."""
        return self._is_circular_lens

    @is_circular_lens.setter
    def is_circular_lens(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Expected is_circular_lens to be type bool, but got {type(value).__name__}.")
        self._is_circular_lens = value

    def extra_repr(self) -> str:
        return super().extra_repr() + f", is_circular_lens={self.is_circular_lens}"
