"""This module defines utility functions for TorchOptics."""

import inspect
from typing import Any

import torch
from torch import Tensor


def initialize_tensor(
    name: str,
    value: Any,
    is_scalar: bool = False,
    is_vector2: bool = False,
    is_complex: bool = False,
    is_integer: bool = False,
    is_positive: bool = False,
    is_non_negative: bool = False,
) -> Tensor:
    """
    Initializes a tensor with validation checks.

    Args:
        name (str): The name of the tensor.
        value (Any): The value to initialize the tensor with.
        is_scalar (bool): If `True`, the tensor is a scalar.
        is_vector2 (bool): If `True`, the tensor is a 2D vector.
        is_complex (bool, optional): If `True`, the tensor is complex. Default: `False`.
        is_integer (bool, optional): If `True`, the tensor is integer. Default: `False`.
        is_positive (bool, optional): If `True`, validates the tensor is positive. Default: `False`.
        is_non_negative (bool, optional): If `True`, validates the tensor is non-negative. Default: `False`.
    """
    if is_complex and is_integer:
        raise ValueError("Expected is_complex and is_integer to be mutually exclusive, but both are True.")
    if is_scalar and is_vector2:
        raise ValueError("Expected is_scalar and is_vector2 to be mutually exclusive, but both are True.")

    value_dtype = torch.as_tensor(value).dtype
    if is_integer and value_dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"Expected {name} to contain integer values, but found non-integer values.")

    dtype = torch.int if is_integer else torch.cdouble if is_complex else torch.double
    tensor = value.clone().to(dtype) if isinstance(value, Tensor) else torch.tensor(value, dtype=dtype)

    if is_scalar:
        if tensor.numel() != 1:
            raise ValueError(f"Expected {name} to be a scalar, but got a tensor with shape {tensor.shape}.")
        tensor = tensor.squeeze()

    if is_vector2:
        if tensor.numel() == 1:  # Convert scalar to 2D vector
            tensor = torch.full((2,), tensor.item())
        if tensor.numel() != 2:
            raise ValueError(
                f"Expected {name} to be a 2D vector, but got a tensor with shape {tensor.shape}."
            )
        tensor = tensor.squeeze()

    if is_positive and not torch.all(tensor > 0):
        raise ValueError(f"Expected {name} to contain positive values, but found non-positive values.")
    if is_non_negative and not torch.all(tensor >= 0):
        raise ValueError(f"Expected {name} to contain non-negative values, but found negative values.")

    return tensor


def validate_tensor_dim(tensor: Tensor, name: str, dim: int) -> None:
    """
    Validates that a PyTorch tensor has the expected shape.

    Args:
        tensor (Tensor): The PyTorch tensor to validate.
        name (str): The name of the tensor, used for error messages.
        shape (tuple): The expected shape of the tensor. Use `-1` as a wildcard
                       to allow any size in that dimension.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Expected '{name}' to be a Tensor, but got {type(tensor).__name__}")
    if tensor.dim() != dim:
        raise ValueError(f"Expected '{name}' to be a {dim}D tensor, but got {tensor.dim()}D")


def copy(obj, **kwargs):
    """
    Creates a copy of an object using its `__init__` parameters.

    Args:
        obj: The object to copy.
        **kwargs: New properties to update.

    Returns:
        A copied object with updated properties.

    Raises:
        ValueError: If the object is missing required instance variables.
    """
    cls = type(obj)
    init_params = [k for k in inspect.signature(cls.__init__).parameters if k != "self"]

    missing_attrs = [k for k in init_params if not hasattr(obj, k)]
    if missing_attrs:
        raise ValueError(
            f"Cannot copy instance of {cls.__name__} because the following required attributes are missing: "
            f"{missing_attrs}"
        )

    new_attrs = {k: getattr(obj, k) for k in init_params}
    new_attrs.update(kwargs)
    return cls(**new_attrs)
