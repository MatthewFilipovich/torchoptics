"""Methods for getting and setting default values for spacing, wavelength, and dtype."""

import torch
from torch import Tensor

from .types import Scalar, Vector2


class Config:
    """Global configuration values for torchoptics."""

    spacing: Tensor | None = None
    wavelength: Tensor | None = None
    dtype: torch.dtype = torch.double


def get_default_spacing() -> Tensor:
    """Get the current default ``spacing`` value."""
    default_spacing = Config.spacing
    if default_spacing is None:
        msg = "Default spacing is not set."
        raise ValueError(msg)
    return default_spacing


def set_default_spacing(value: Vector2) -> None:
    """Set the default ``spacing`` value.

    Args:
        value (Vector2): The default spacing.

    Example:
        >>> torchoptics.set_default_spacing((10e-6, 10e-6))

    """
    from .utils import initialize_tensor

    Config.spacing = initialize_tensor("spacing", value, is_vector2=True, is_positive=True)


def get_default_wavelength() -> Tensor:
    """Get the current default ``wavelength`` value."""
    default_wavelength = Config.wavelength
    if default_wavelength is None:
        msg = "Default wavelength is not set."
        raise ValueError(msg)
    return default_wavelength


def set_default_wavelength(value: Scalar) -> None:
    """Set the default ``wavelength`` value.

    Args:
        value (Scalar): The default wavelength.

    Example:
        >>> torchoptics.set_default_wavelength(700e-6)

    """
    from .utils import initialize_tensor

    Config.wavelength = initialize_tensor("wavelength", value, is_scalar=True, is_positive=True)


def get_default_dtype() -> torch.dtype:
    """Get the current default ``dtype`` value."""
    return Config.dtype


def set_default_dtype(value: torch.dtype) -> None:
    """Set the default ``dtype`` value.

    Args:
        value (torch.dtype): The default dtype.

    Example:
        >>> torchoptics.set_default_dtype(torch.float32)

    """
    if not isinstance(value, torch.dtype):
        msg = f"Expected value to be a torch.dtype, but got {type(value).__name__}."
        raise TypeError(msg)
    if value not in (torch.float32, torch.float64):
        msg = f"Expected value to be torch.float32 or torch.float64, but got {value}."
        raise ValueError(msg)
    Config.dtype = value
