"""This module defines the IdentityElement element."""

from ..fields import Field
from .elements import Element

__all__ = ["IdentityElement"]


class IdentityElement(Element):
    """
    Identity element.

    The identity element does not modify the input field. It is useful as a placeholder element in
    :class:`torchoptics.System`.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def forward(self, field: Field) -> Field:
        """Returns the input field without modification."""
        self.validate_field(field)
        return field
