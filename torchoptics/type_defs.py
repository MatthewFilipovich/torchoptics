"""This module defines type aliases for the torchoptics package."""

from typing import Sequence, Union

from torch import Tensor

from .param import Param

__all__ = ["Scalar", "Vector2"]

Scalar = Union[int, float, bool, complex, bytes, Tensor, Param]
Vector2 = Union[int, float, bool, complex, bytes, Tensor, Param, list, tuple, Sequence]
