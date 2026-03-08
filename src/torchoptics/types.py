"""This module defines type aliases for the torchoptics package."""

from typing import Sequence, Union

from torch import Tensor
from typing_extensions import TypeAlias

Int: TypeAlias = Union[int, Tensor]
Scalar: TypeAlias = Union[int, float, Tensor]
Vector2: TypeAlias = Union[int, float, Tensor, Sequence]
