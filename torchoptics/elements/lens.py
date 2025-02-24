"""This module defines the Lens element."""

from typing import Optional

from torch import Tensor

from ..profiles import lens
from ..type_defs import Scalar, Vector2
from .elements import PolychromaticModulationElement

__all__ = ["Lens"]


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
        is_circular_lens (bool): If `True`, the lens is circular and the phase profile is set to zero outside
            the lens diameter, otherwise lens is square. Default: `True`.
    """

    focal_length: Tensor

    def __init__(
        self,
        shape: Vector2,
        focal_length: Scalar,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
        is_circular_lens: bool = True,
    ) -> None:
        super().__init__(shape, z, spacing, offset)
        self.register_optics_property("focal_length", focal_length, is_scalar=True)
        self.is_circular_lens = is_circular_lens

    def modulation_profile(self, wavelength: Optional[Scalar] = None) -> Tensor:
        return lens(self.shape, self.focal_length, wavelength, self.spacing, None, self.is_circular_lens)

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
