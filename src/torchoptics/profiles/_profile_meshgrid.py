from torch import Tensor

from ..planar_grid import PlanarGrid
from ..types import Vector2


def profile_meshgrid(
    shape: Vector2,
    spacing: Vector2 | None,
    offset: Vector2 | None,
) -> tuple[Tensor, Tensor]:
    """Generate a meshgrid for a 2D profile with inverted offset."""
    planar_grid = PlanarGrid(shape, spacing=spacing, offset=offset)
    planar_grid.offset = -planar_grid.offset  # Invert the offset for meshgrid
    return planar_grid.meshgrid()
