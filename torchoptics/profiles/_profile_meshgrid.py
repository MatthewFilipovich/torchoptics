from typing import Optional

from torch import Tensor

from ..planar_grid import PlanarGrid
from ..type_defs import Vector2


def profile_meshgrid(
    shape: Vector2,
    spacing: Optional[Vector2],
    offset: Optional[Vector2],
) -> tuple[Tensor, Tensor]:
    "Generate a meshgrid for a 2D profile with inverted offset."
    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    planar_grid.offset = -planar_grid.offset  # Invert the offset for meshgrid
    return planar_grid.meshgrid()
