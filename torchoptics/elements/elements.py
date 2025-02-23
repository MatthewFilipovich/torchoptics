"""This module contains the base classes for the optical elements."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch import Tensor

from ..fields import Field, PolarizedField
from ..planar_geometry import PlanarGeometry
from ..type_defs import Scalar

__all__ = ["Element", "ModulationElement", "PolarizedModulationElement"]


class Element(PlanarGeometry):  # pylint: disable=abstract-method
    """
    Base class for optical elements.

    The :meth:`forward()` method must be implemented by the subclass.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def validate_field(self, field: Field) -> None:
        """
        Validates that the field has the same geometry as the element.

        Args:
            field (Field): The field to validate.
        """
        if not isinstance(field, Field):
            raise TypeError(f"Expected field to be a Field, but got {type(field).__name__}.")

        if not self.is_same_geometry(field):
            raise ValueError(
                f"Expected field to have same geometry as element:"
                f"\nField geometry:   {field.geometry_str()}"
                f"\nElement geometry: {self.geometry_str()}"
            )


class ModulationElement(Element, ABC):
    """
    Base class for elements that modulate the field.

    The :attr:`modulation_profile` property must be implemented by the subclass.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def forward(self, field: Field) -> Field:
        """
        Modulates the field.

        Args:
            field (Field): The field to modulate.

        Returns:
            Field: The modulated field."""
        self.validate_field(field)
        return field.modulate(self.modulation_profile())

    def visualize(self, **kwargs) -> Any:
        """
        Visualizes the modulation profile.

        Args:
            **kwargs: Additional keyword arguments for visualization.
        """
        kwargs.update({"symbol": r"$\mathcal{M}$"})
        return self._visualize(self.modulation_profile(), **kwargs)

    @abstractmethod
    def modulation_profile(self) -> Tensor:
        """Returns the modulation profile."""


class PolychromaticModulationElement(Element):
    """
    Base class for elements that modulate the field based on the wavelength.

    The :meth:`modulation_profile` property must be implemented by the subclass.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def forward(self, field: Field) -> Field:
        """
        Modulates the field.

        Args:
            field (Field): The field to modulate.

        Returns:
            Field: The modulated field."""
        self.validate_field(field)
        return field.modulate(self.modulation_profile(field.wavelength))

    def visualize(self, wavelength: Optional[Scalar] = None, **kwargs) -> Any:
        """
        Visualizes the modulation profile.

        Args:
            **kwargs: Additional keyword arguments for visualization.
        """
        kwargs.update({"symbol": r"$\mathcal{M}$"})
        return self._visualize(self.modulation_profile(wavelength), **kwargs)

    @abstractmethod
    def modulation_profile(self, wavelength: Optional[Scalar] = None) -> Tensor:
        """Returns the modulation profile for the given wavelength."""


class PolarizedModulationElement(Element):
    """
    Base class for elements that modulate the polarized field.

    The :attr:`polarized_modulation_profile` property must be implemented by the subclass.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def forward(self, field: PolarizedField) -> PolarizedField:
        """
        Modulates the polarized field.

        Args:
            field (PolarizedField): The field to modulate.

        Returns:
            PolarizedField: The modulated polarized field."""
        self.validate_field(field)
        return field.polarized_modulate(self.polarized_modulation_profile())

    def visualize(self, *index: int, **kwargs) -> Any:
        """
        Visualizes the polarized modulation profile.

        Args:
            *index (int): Index of the tensor to visualize.
            **kwargs: Additional keyword arguments for visualization.
        """
        kwargs.update({"symbol": "$J$"})
        return self._visualize(self.polarized_modulation_profile(), index, **kwargs)

    @abstractmethod
    def polarized_modulation_profile(self) -> Tensor:
        """Returns the polarized modulation profile."""
