"""Shape profile generation functions."""

import torch
from torch import Tensor

from ..config import get_default_dtype
from ..types import Scalar, Vector2
from ..utils import initialize_tensor
from ._profile_meshgrid import profile_meshgrid


def checkerboard(
    shape: Vector2,
    tile_length: Vector2,
    num_tiles: Vector2,
    spacing: Vector2 | None = None,
    offset: Vector2 | None = None,
) -> Tensor:
    """Generate a checkerboard pattern.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        tile_length (Vector2): The side lengths of the checkerboard tiles.
        num_tiles (Vector2): Number of tiles along the planar dimensions.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Offset coordinates of the pattern. Default: `(0, 0)`.

    Returns:
        Tensor: The generated checkerboard pattern with internal padding.

    """
    tile_length = initialize_tensor("tile_length", tile_length, is_vector2=True, is_positive=True)
    num_tiles = initialize_tensor("num_tiles", num_tiles, is_vector2=True, is_integer=True, is_positive=True)
    x, y = profile_meshgrid(shape, spacing, offset)

    x_tile = (x + (tile_length[0] / 2 if num_tiles[0] % 2 == 1 else 0)) // tile_length[0]
    y_tile = (y + (tile_length[1] / 2 if num_tiles[1] % 2 == 1 else 0)) // tile_length[1]

    pattern = (1 + x_tile + y_tile) % 2
    pattern[x.abs() * 2 >= tile_length[0] * num_tiles[0]] = 0
    pattern[y.abs() * 2 >= tile_length[1] * num_tiles[1]] = 0

    return pattern


def circle(
    shape: Vector2,
    radius: Scalar,
    spacing: Vector2 | None = None,
    offset: Vector2 | None = None,
) -> Tensor:
    """Generate a circular profile.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        radius (Scalar): The radius of the circle.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the profile. Default: `(0, 0)`.

    Returns:
        Tensor: The generated circular profile.

    """
    radius = initialize_tensor("radius", radius, is_scalar=True, is_positive=True)
    x, y = profile_meshgrid(shape, spacing, offset)
    r = torch.sqrt(x**2 + y**2)
    return (r <= radius).to(get_default_dtype())


def rectangle(
    shape: Vector2,
    side: Vector2,
    spacing: Vector2 | None = None,
    offset: Vector2 | None = None,
) -> Tensor:
    """Generate a rectangle profile.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        side (Vector2): The two side lengths of the rectangle.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the profile. Default: `(0, 0)`.

    Returns:
        Tensor: The generated rectangle profile.

    """
    side = initialize_tensor("side", side, is_vector2=True, is_positive=True)
    x, y = profile_meshgrid(shape, spacing, offset)
    return ((x.abs() <= side[0] / 2) & (y.abs() <= side[1] / 2)).to(get_default_dtype())


def square(
    shape: Vector2,
    side: Scalar,
    spacing: Vector2 | None = None,
    offset: Vector2 | None = None,
) -> Tensor:
    """Generate a square profile.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        side (Scalar): The side length of the square.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the profile. Default: `(0, 0)`.

    Returns:
        Tensor: The generated square profile.

    """
    return rectangle(shape, (side, side), spacing, offset)


def triangle(
    shape: Vector2,
    base: Scalar,
    height: Scalar,
    spacing: Vector2 | None = None,
    offset: Vector2 | None = None,
    theta: Scalar = 0,
) -> Tensor:
    """Generate a triangular profile.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        base (Scalar): The base length of the triangle.
        height (Scalar): The height of the triangle.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the profile. Default: `(0, 0)`.
        theta (Scalar): The angle of rotation of the triangle in radians. Default: `0`.

    Returns:
        Tensor: The generated triangular profile.

    """
    base = initialize_tensor("base", base, is_scalar=True, is_positive=True)
    height = initialize_tensor("height", height, is_scalar=True, is_positive=True)
    theta = initialize_tensor("theta", theta, is_scalar=True)
    x, y = profile_meshgrid(shape, spacing, offset)

    theta -= torch.pi / 2
    x_rot = x * torch.cos(theta) - y * torch.sin(theta)
    y_rot = x * torch.sin(theta) + y * torch.cos(theta)

    return (
        (y_rot >= -height / 2)
        & (y_rot <= height / 2)
        & (x_rot.abs() <= base / 2)
        & (y_rot <= height / 2 - (2 * height / base) * x_rot.abs())
    ).to(get_default_dtype())
