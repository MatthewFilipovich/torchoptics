"""This module defines methods for getting and setting default values for spacing and wavelength."""

from typing import Optional

from torch import Tensor

from .functional import initialize_tensor
from .type_defs import Scalar, Vector2

__all__ = ["get_default_spacing", "set_default_spacing", "get_default_wavelength", "set_default_wavelength"]


class Config:
    """Global configuration values for torchoptics."""

    spacing: Optional[Tensor] = None
    wavelength: Optional[Tensor] = None


def get_default_spacing() -> Tensor:
    """Gets the current default ``spacing`` value."""
    default_spacing = Config.spacing
    if default_spacing is None:
        raise ValueError("Default spacing is not set.")
    return default_spacing


def set_default_spacing(value: Vector2) -> None:
    """
    Sets the default ``spacing`` value.

    Args:
        value (Vector2): The default spacing.

    Example:
        >>> torchoptics.set_default_spacing((10e-6, 10e-6))
    """
    Config.spacing = initialize_tensor("spacing", value, (2,), validate_positive=True, fill_value=True)


def get_default_wavelength() -> Tensor:
    """Gets the current default ``wavelength`` value."""
    default_wavelength = Config.wavelength
    if default_wavelength is None:
        raise ValueError("Default wavelength is not set.")
    return default_wavelength


def set_default_wavelength(value: Scalar) -> None:
    """
    Sets the default ``wavelength`` value.

    Args:
        value (Scalar): The default wavelength.

    Example:
        >>> torchoptics.set_default_wavelength(700e-6)
    """
    Config.wavelength = initialize_tensor("wavelength", value, (), validate_positive=True)
