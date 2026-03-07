"""This module defines the System class."""

from typing import Iterator, Optional, Union, overload

from torch.nn import Module

from .elements import Element, IdentityElement
from .fields import Field
from .planar_grid import PlanarGrid
from .type_defs import Scalar, Vector2


class System(Module):
    """
    Optical system of elements similar to :class:`torch.nn.Sequential`.

    The system consists of a sequence of optical elements, ordered by their ``z`` positions.
    When a :class:`~torchoptics.Field` is passed to :meth:`forward`, it is propagated through the system:
    each element applies its own transformation via :meth:`~torchoptics.elements.Element.forward`.
    The output from the final element is returned.

    Field measurements at arbitrary planes can be performed using :meth:`measure`, :meth:`measure_at_z`, or
    :meth:`measure_at_plane`.

    Indexing with ``system[i]`` returns the i-th optical element. Slicing, e.g. ``system[i:j]``,
    returns a new :class:`System` containing the selected elements.

    Example:
        Create a 4f system consisting of two lenses::

            system = System(
                Lens(shape, focal_length, z=1 * focal_length),
                Lens(shape, focal_length, z=3 * focal_length),
            ).to(device)

            # Measure the field at the 4f plane
            output_field = system.measure_at_z(input_field, z=4 * focal_length)

    Args:
        *elements (Element): Optical elements in the system.
    """

    def __init__(self, *elements: Element) -> None:
        super().__init__()
        for i, element in enumerate(elements):
            if not isinstance(element, Element):
                raise TypeError(f"Expected element {i} to be an Element, but got {type(element).__name__}.")
            self.add_module(str(i), element)
        self._elements = tuple(elements)

    def __iter__(self) -> Iterator[Element]:
        return iter(self.elements)

    @overload
    def __getitem__(self, index: int) -> Element: ...

    @overload
    def __getitem__(self, index: slice) -> "System": ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Element, "System"]:
        if isinstance(index, slice):
            return self.__class__(*self.elements[index])
        return self.elements[index]

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def elements(self) -> tuple[Element, ...]:
        """Returns the elements in the system."""
        return self._elements

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

    def measure_at_plane(self, field: Field, plane: PlanarGrid, **prop_kwargs) -> Field:
        """
        Propagates the field through the system to a plane defined by a :class:`PlanarGrid` object.

        Args:
            field (Field): Input field.
            plane (PlanarGrid): Plane grid.
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
            if not isinstance(last_element, Element):
                raise TypeError(
                    f"Expected last_element to be an Element, but got {type(last_element).__name__}."
                )
            if last_element.z < field.z:
                raise ValueError(f"Field z ({field.z}) is greater than last element z ({last_element.z}).")

            elements_in_path = [element for element in elements_in_path if element.z <= last_element.z]

            # Remove trailing IdentityElement before appending last_element
            if elements_in_path and isinstance(elements_in_path[-1], IdentityElement):
                elements_in_path.pop()

            elements_in_path.append(last_element)

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
