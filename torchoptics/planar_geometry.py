"""This module defines the PlanarGeometry class."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from .config import get_default_spacing
from .functional import initialize_tensor, meshgrid2d
from .optics_module import OpticsModule
from .type_defs import Scalar, Vector2
from .visualization import visualize_tensor

__all__ = ["PlanarGeometry"]


class PlanarGeometry(OpticsModule):  # pylint: disable=abstract-method
    """
    Base class for TorchOptics classes with 2D planar geometries.

    This class defines objects with planar geometries perpendicular to the z-axis. It includes methods
    for calculating various properties of the planar geometry.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    def __init__(
        self,
        shape: Vector2,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        super().__init__()
        self._shape = torch.Size(
            initialize_tensor("shape", shape, (2,), is_integer=True, validate_positive=True, fill_value=True)
        )
        self.register_optics_property("z", z, ())
        self.register_optics_property(
            "spacing",
            get_default_spacing() if spacing is None else spacing,
            (2,),
            validate_positive=True,
            fill_value=True,
        )
        self.register_optics_property("offset", (0, 0) if offset is None else offset, (2,))

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of the plane."""
        return self._shape

    @property
    def geometry(self) -> dict[str, Any]:
        """Returns a dictionary containing ``shape``, ``z``, ``spacing``, and ``offset``."""
        return {"shape": self.shape, "z": self.z, "spacing": self.spacing, "offset": self.offset}

    def extra_repr(self) -> str:
        """Returns the extra representation of the class."""
        strings = [f"shape=({self.shape[0]}, {self.shape[1]})"]
        for name in self._optics_property_configs:
            value = getattr(self, name)
            if value.numel() == 1:
                strings.append(f"{name}={value.item():.2e}")
            elif value.numel() == 2:
                strings.append(f"{name}=({value[0].item():.2e}, {value[1].item():.2e})")
        return ", ".join(strings)

    def cell_area(self) -> Tensor:
        """Returns the area between adjacent grid points."""
        return self.spacing[0] * self.spacing[1]

    def length(self, use_grid_points=False) -> Tensor:
        """
        Returns the length of the plane along the planar dimensions.

        Args:
            use_grid_points (bool): If `True`, returns the length between the first and last grid points along
                the planar dimensions. Otherwise, returns the length between the edges of the first and last
                grid cells. Default: `False`.
        """
        shape_tensor = self.spacing.new_tensor(self.shape)
        return self.spacing * (shape_tensor - 1) if use_grid_points else self.spacing * shape_tensor

    def bounds(self, use_grid_points=False) -> Tensor:
        """
        Returns the position of the plane boundaries along the planar dimensions.

        Args:
            use_grid_points (bool): If `True`, returns the position of the first and last grid points along
                the planar dimensions. Otherwise, returns the position of the edges of the first and last
                grid cells. Default: `False`.
        """
        half_length = self.length(use_grid_points) / 2
        bounds = (
            self.offset[0] - half_length[0],
            self.offset[0] + half_length[0],
            self.offset[1] - half_length[1],
            self.offset[1] + half_length[1],
        )
        return torch.stack(bounds)

    def meshgrid(self) -> tuple[Tensor, Tensor]:
        """Returns a 2D meshgrid of the grid points along the plane."""
        return meshgrid2d(self.bounds(use_grid_points=True), self.shape)

    def is_same_geometry(self, other: PlanarGeometry) -> bool:
        """
        Checks if the geometry is the same as another :class:`PlanarGeometry` instance.

        Args:
            other (PlanarGeometry): Another instance of PlanarGeometry to compare with.
        """
        return (
            self.shape == other.shape
            and torch.equal(self.z, other.z)
            and torch.equal(self.spacing, other.spacing)
            and torch.equal(self.offset, other.offset)
        )

    def geometry_str(self) -> str:
        """Returns a string representation of the geometry properties."""
        shape_str = f"({self.shape[0]}, {self.shape[1]})"
        spacing_str = f"({self.spacing[0].item():.2e}, {self.spacing[1].item():.2e})"
        offset_str = f"({self.offset[0].item():.2e}, {self.offset[1].item():.2e})"
        return f"shape={shape_str}, z={self.z.item():.2e}, spacing={spacing_str}, offset={offset_str}"

    def _visualize(self, data: Tensor, index: tuple = (), bounds: bool = False, **kwargs) -> Any:
        """Visualizes the data tensor."""
        if bounds:
            kwargs.update({"extent": torch.cat((self.bounds()[2:], self.bounds()[:2])).cpu().detach()})
        return visualize_tensor(data[index + (slice(None), slice(None))], **kwargs)