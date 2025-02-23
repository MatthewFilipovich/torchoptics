"""This module defines methods for getting and setting default values for spacing and wavelength."""

from typing import Optional

from torch import Tensor

from .type_defs import Scalar, Vector2
from .utils import initialize_tensor

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
    Config.spacing = initialize_tensor("spacing", value, is_vector2=True, is_positive=True)


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
    Config.wavelength = initialize_tensor("wavelength", value, is_scalar=True, is_positive=True)


def spacing_or_default(spacing: Optional[Vector2]) -> Tensor:
    """Gets the spacing or the default value if ``spacing`` is ``None``."""
    return (
        get_default_spacing()
        if spacing is None
        else initialize_tensor("spacing", spacing, is_vector2=True, is_positive=True)
    )


def wavelength_or_default(wavelength: Optional[Scalar]) -> Tensor:
    """Gets the wavelength or the default value if ``wavelength`` is ``None``."""
    return (
        get_default_wavelength()
        if wavelength is None
        else initialize_tensor("wavelength", wavelength, is_scalar=True, is_positive=True)
    )
