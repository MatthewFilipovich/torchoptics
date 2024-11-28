"""This module defines functions to generate profiles with different shapes."""

from typing import Optional

import torch
from torch import Tensor

from ..functional import initialize_tensor
from ..planar_geometry import PlanarGeometry
from ..type_defs import Vector2

__all__ = ["checkerboard", "circle", "rectangle", "square"]


def checkerboard(
    shape: Vector2,
    tile_length: Vector2,
    num_tiles: Vector2,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
) -> Tensor:
    """
    Generates a checkerboard pattern.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        tile_length (Vector2): The side lengths of the checkerboard tiles.
        num_tiles (Vector2): Number of tiles along the planar dimensions.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Offset coordinates of the pattern. Default: `(0, 0)`.

    Returns:
        Tensor: The generated checkerboard pattern with internal padding.
    """
    x, y = PlanarGeometry(shape, 0, spacing, offset).meshgrid()
    tile_length = initialize_tensor("tile_length", tile_length, (2,), validate_positive=True, fill_value=True)
    num_tiles = initialize_tensor(
        "num_tiles", num_tiles, (2,), is_integer=True, validate_positive=True, fill_value=True
    )

    x_tile = (x + (tile_length[0] / 2 if num_tiles[0] % 2 == 1 else 0)) // tile_length[0]
    y_tile = (y + (tile_length[1] / 2 if num_tiles[1] % 2 == 1 else 0)) // tile_length[1]

    pattern = (1 + x_tile + y_tile) % 2
    pattern[x.abs() * 2 >= tile_length[0] * num_tiles[0]] = 0
    pattern[y.abs() * 2 >= tile_length[1] * num_tiles[1]] = 0

    return pattern


def circle(
    shape: Vector2, radius: float, spacing: Optional[Vector2] = None, offset: Optional[Vector2] = None
) -> Tensor:
    """
    Generates a circular profile.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        radius (float): The radius of the circle.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the profile. Default: `(0, 0)`.

    Returns:
        Tensor: The generated circular profile.
    """
    x, y = PlanarGeometry(shape, 0, spacing, offset).meshgrid()
    r = torch.sqrt(x**2 + y**2)
    return r <= radius  # type: ignore[attr-defined]


def rectangle(
    shape: Vector2, side: Vector2, spacing: Optional[Vector2] = None, offset: Optional[Vector2] = None
) -> Tensor:
    """
    Generates a rectangle profile.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        side (Vector2): The two side lengths of the rectangle.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the profile. Default: `(0, 0)`.

    Returns:
        Tensor: The generated rectangle profile.
    """
    x, y = PlanarGeometry(shape, 0, spacing, offset).meshgrid()
    side = initialize_tensor("side", side, (2,), validate_positive=True, fill_value=True)
    return (x.abs() <= side[0] / 2) & (y.abs() <= side[1] / 2)  # type: ignore


def square(
    shape: Vector2, side: float, spacing: Optional[Vector2] = None, offset: Optional[Vector2] = None
) -> Tensor:
    """
    Generates a square profile.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        side (float): The side length of the square.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the profile. Default: `(0, 0)`.

    Returns:
        Tensor: The generated square profile.
    """
    return rectangle(shape, (side, side), spacing, offset)
