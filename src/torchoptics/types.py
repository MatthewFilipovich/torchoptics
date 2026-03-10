"""This module defines type aliases for the torchoptics package."""

from collections.abc import Sequence
from typing import TypeAlias

from torch import Tensor

Int: TypeAlias = int | Tensor
Scalar: TypeAlias = int | float | Tensor
Vector2: TypeAlias = int | float | Tensor | Sequence
