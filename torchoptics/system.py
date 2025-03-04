"""This module defines the System class."""

from typing import Optional

from torch.nn import Module, ModuleList

from .elements import Element, IdentityElement
from .fields import Field
from .planar_geometry import PlanarGeometry
from .type_defs import Scalar, Vector2

__all__ = ["System"]


class System(Module):
    """
    System of optical elements similar to the :class:`torch.nn.Sequential` module.

    The system is defined by a sequence of optical elements which are sorted by their ``z`` position.
    The :meth:`forward()` method accepts a :class:`Field` object as input. The field is
    propagated to the first element in the system which processes it using its ``forward()`` method.
    The field is then propagated to the next element in the system and so on, finally returning the
    field after it has been processed by the last element in the system.

    Example:
        Initialize a 4f optical system with two lenses::

            import torch
            import torchoptics
            from torchoptics import Field, System
            from torchoptics.elements import Lens

            # Set simulation properties
            shape = 1000  # Number of grid points in each dimension
            spacing = 10e-6  # Spacing between grid points (m)
            wavelength = 700e-9  # Field wavelength (m)
            focal_length = 200e-3  # Lens focal length (m)

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Configure torchoptics default properties
            torchoptics.set_default_spacing(spacing)
            torchoptics.set_default_wavelength(wavelength)

            # Define 4f optical system with two lenses
            system = System(
                Lens(shape, focal_length, z=1 * focal_length),
                Lens(shape, focal_length, z=3 * focal_length),
            ).to(device)

    Args:
        *elements (Element): Optical elements in the system.
    """

    def __init__(self, *elements: Element) -> None:
        super().__init__()
        self.elements = ModuleList(elements)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return len(self.elements)

    def forward(self, field: Field, **prop_kwargs) -> Field:
        """
        Propagates the field through the system.

        Args:
            field (Field): Input field.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating through the system."""
        return self._forward(field, None, **prop_kwargs)

    def measure(
        self,
        field: Field,
        shape: Vector2,
        z: Scalar,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
        **prop_kwargs,
    ) -> Field:
        """
        Propagates the field through the system to a plane defined by the input parameters.

        Args:
            field (Field): Input field.
            shape (Vector2): Number of grid points along the planar dimensions.
            z (Scalar): Position along the z-axis.
            spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default:
                if `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
            offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating to the plane.
        """
        last_element = IdentityElement(shape, z, spacing, offset).to(field.data.device)
        return self._forward(field, last_element, **prop_kwargs)

    def measure_at_z(self, field: Field, z: Scalar, **prop_kwargs) -> Field:
        """
        Propagates the field through the system to a plane at a specific z position.

        The plane has the same ``shape``, ``spacing``, and ``offset`` as the input field.

        Args:
            field (Field): Input field.
            z (Scalar): Position along the z-axis.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating to the plane.
        """
        return self.measure(field, field.shape, z, field.spacing, field.offset, **prop_kwargs)

    def measure_at_plane(self, field: Field, plane: PlanarGeometry, **prop_kwargs) -> Field:
        """
        Propagates the field through the system to a plane defined by a :class:`PlanarGeometry` object.

        Args:
            field (Field): Input field.
            plane (PlanarGeometry): Plane geometry.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating to the plane.
        """
        return self.measure(field, plane.shape, plane.z, plane.spacing, plane.offset, **prop_kwargs)

    def sorted_elements(self) -> tuple[Element, ...]:
        """Returns the elements sorted by their z position."""
        return tuple(sorted(self.elements, key=lambda element: element.z.item()))

    def elements_in_field_path(self, field: Field, last_element: Optional[Element]) -> tuple[Element, ...]:
        """
        Returns the elements along the field path.

        Args:
            field (Field): Input field.
            last_element (Optional[Element]): Last element of the system.

        Returns:
            tuple[Element]: Elements along the field path.
        """
        elements_in_path = [element for element in self.sorted_elements() if field.z <= element.z]

        if last_element:
            if last_element.z < field.z:
                raise ValueError(f"Field z ({field.z}) is greater than last element z ({last_element.z}).")

            elements_in_path = [element for element in elements_in_path if element.z <= last_element.z]

            # Remove trailing IdentityElement before appending last_element
            if elements_in_path and isinstance(elements_in_path[-1], IdentityElement):
                elements_in_path.pop()

            elements_in_path.append(last_element)

        if not elements_in_path:
            raise ValueError("No elements found in the field path.")
        if not all(isinstance(element, Element) for element in elements_in_path):
            raise TypeError("All elements in the field path must be instances of Element.")

        return tuple(elements_in_path)

    def _forward(self, field: Field, last_element: Optional[Element], **prop_kwargs) -> Field:
        """Propagates the field through the system to the last element, if provided."""
        elements = self.elements_in_field_path(field, last_element)

        for i, element in enumerate(elements):
            field = field.propagate_to_plane(element, **prop_kwargs)
            field = element(field)

            if not isinstance(field, Field) and i < len(elements) - 1:
                raise TypeError(
                    f"Expected all elements in the field path, except for the last, to return a Field. "
                    f"Element at index {i} ({type(element).__name__}) returned {type(field).__name__}."
                )

        return field
