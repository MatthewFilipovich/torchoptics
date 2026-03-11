"""Utility functions for TorchOptics."""

from typing import Any

import torch
from torch import Tensor

from .config import get_default_dtype, get_default_spacing, get_default_wavelength
from .types import Scalar, Vector2


def initialize_tensor(
    name: str,
    value: Any,
    *,
    is_scalar: bool = False,
    is_vector2: bool = False,
    is_complex: bool = False,
    is_integer: bool = False,
    is_positive: bool = False,
    is_non_negative: bool = False,
) -> Tensor:
    """Initialize a tensor with validation checks.

    Args:
        name (str): The name of the tensor.
        value (Any): The value to initialize the tensor with.
        is_scalar (bool): If `True`, the tensor is a scalar.
        is_vector2 (bool): If `True`, the tensor is a 2D vector.
        is_complex (bool): If `True`, the tensor is complex. Default: `False`.
        is_integer (bool): If `True`, the tensor is integer. Default: `False`.
        is_positive (bool): If `True`, validates the tensor is positive. Default: `False`.
        is_non_negative (bool): If `True`, validates the tensor is non-negative. Default: `False`.

    """
    if is_complex and is_integer:
        msg = "Expected is_complex and is_integer to be mutually exclusive, but both are True."
        raise ValueError(msg)
    if is_scalar and is_vector2:
        msg = "Expected is_scalar and is_vector2 to be mutually exclusive, but both are True."
        raise ValueError(msg)

    value_dtype = torch.as_tensor(value).dtype
    if is_integer and value_dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        msg = f"Expected {name} to contain integer values, but found non-integer values."
        raise ValueError(msg)

    default_dtype = get_default_dtype()

    if is_integer:
        dtype = torch.int64
    elif is_complex:
        dtype = torch.complex128 if default_dtype == torch.float64 else torch.complex64
    else:
        dtype = default_dtype
    tensor = value.clone().to(dtype) if isinstance(value, Tensor) else torch.tensor(value, dtype=dtype)

    if is_scalar:
        if tensor.numel() != 1:
            msg = f"Expected {name} to be a scalar, but got a tensor with shape {tensor.shape}."
            raise ValueError(msg)
        tensor = tensor.squeeze()

    if is_vector2:
        if tensor.numel() == 1:  # Convert scalar to 2D vector
            tensor = torch.full((2,), tensor.item())
        if tensor.numel() != 2:
            msg = f"Expected {name} to be a 2D vector, but got a tensor with shape {tensor.shape}."
            raise ValueError(msg)
        tensor = tensor.squeeze()

    if is_positive and not torch.all(tensor > 0):
        msg = f"Expected {name} to contain positive values, but found non-positive values."
        raise ValueError(msg)
    if is_non_negative and not torch.all(tensor >= 0):
        msg = f"Expected {name} to contain non-negative values, but found negative values."
        raise ValueError(msg)

    return tensor


def initialize_shape(shape: Vector2) -> tuple[int, int]:
    """Initialize a 2D shape tensor with validation checks.

    Args:
        shape (Vector2): The shape to initialize.

    """
    shape_tensor = initialize_tensor("shape", shape, is_vector2=True, is_integer=True, is_positive=True)
    return (shape_tensor[0].item(), shape_tensor[1].item())  # type: ignore[return-value]


def spacing_or_default(spacing: Vector2 | None) -> Tensor:
    """Get the spacing or the default value if ``spacing`` is ``None``."""
    return (
        get_default_spacing()
        if spacing is None
        else initialize_tensor("spacing", spacing, is_vector2=True, is_positive=True)
    )


def wavelength_or_default(wavelength: Scalar | None) -> Tensor:
    """Get the wavelength or the default value if ``wavelength`` is ``None``."""
    return (
        get_default_wavelength()
        if wavelength is None
        else initialize_tensor("wavelength", wavelength, is_scalar=True, is_positive=True)
    )


def validate_tensor_ndim(tensor: Tensor, name: str, ndim: int) -> None:
    """Validate that a PyTorch tensor has the expected number of dimensions.

    Args:
        tensor (Tensor): The PyTorch tensor to validate.
        name (str): The name of the tensor, used for error messages.
        ndim (int): The expected number of dimensions.

    """
    if not isinstance(tensor, Tensor):
        msg = f"Expected '{name}' to be a Tensor, but got {type(tensor).__name__}"
        raise TypeError(msg)
    if tensor.ndim != ndim:
        msg = f"Expected '{name}' to be a {ndim}D tensor, but got {tensor.ndim}D"
        raise ValueError(msg)


def validate_tensor_min_ndim(tensor: Tensor, name: str, min_ndim: int) -> None:
    """Validate that a PyTorch tensor has at least a minimum number of dimensions.

    Args:
        tensor (Tensor): The PyTorch tensor to validate.
        name (str): The name of the tensor, used in error messages.
        min_ndim (int): The minimum number of dimensions required.

    Raises:
        TypeError: If the input is not a Tensor.
        ValueError: If the tensor does not meet the minimum dimension requirement.

    """
    if not isinstance(tensor, Tensor):
        msg = f"Expected '{name}' to be a Tensor, but got {type(tensor).__name__}."
        raise TypeError(msg)

    if tensor.ndim < min_ndim:
        msg = f"Expected '{name}' to have at least {min_ndim} dimensions, but got {tensor.ndim}."
        raise ValueError(msg)
