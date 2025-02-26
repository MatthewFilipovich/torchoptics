"""This module defines type aliases for the torchoptics package."""

from typing import Sequence, Union

from torch import Tensor

__all__ = ["Scalar", "Vector2"]

Int = Union[int, Tensor]
Scalar = Union[int, float, Tensor]
Vector2 = Union[int, float, Tensor, Sequence]
