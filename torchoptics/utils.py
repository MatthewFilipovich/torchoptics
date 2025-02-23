"""This module defines utility functions for TorchOptics."""

from torch import Tensor


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
