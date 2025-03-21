"""This module defines the Field and SpatialCoherence classes."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from .config import wavelength_or_default
from .functional import calculate_centroid, calculate_std, get_coherence_evolution, inner2d, outer2d
from .planar_grid import PlanarGrid
from .propagation import propagator
from .type_defs import Scalar, Vector2
from .utils import copy, validate_tensor_min_ndim

__all__ = ["Field", "SpatialCoherence"]


class Field(PlanarGrid):  # pylint: disable=abstract-method
    """
    Optical field class.

    Args:
        data (Tensor): The complex-valued field data.
        wavelength (Scalar, optional): The wavelength of the field. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    DATA_MIN_NDIM = 2
    POLARIZATION_DIM = -3
    data: Tensor
    wavelength: Tensor

    def __init__(
        self,
        data: Tensor,
        wavelength: Optional[Scalar] = None,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:

        validate_tensor_min_ndim(data, "data", self.DATA_MIN_NDIM)
        super().__init__(data.shape[-2:], z, spacing, offset)
        self.register_optics_property("data", data, is_complex=True)
        self.register_optics_property(
            "wavelength", wavelength_or_default(wavelength), is_scalar=True, is_positive=True
        )

    def intensity(self) -> Tensor:
        """Returns the intensity of the field."""
        return self.data.abs().square()

    def power(self) -> Tensor:
        """Returns the total power of the field calculated by integrating the intensity over the plane."""
        return self.intensity().sum(dim=(-1, -2)) * self.cell_area()

    def centroid(self) -> Tensor:
        """Returns the centroid of the intensity."""
        return calculate_centroid(self.intensity(), self.meshgrid())

    def std(self) -> Tensor:
        """Returns the standard deviation of the intensity."""
        return calculate_std(self.intensity(), self.meshgrid())

    def propagate(
        self,
        shape: Vector2,
        z: Scalar,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
        propagation_method: str = "AUTO",
        asm_pad_factor: Vector2 = 2,
        interpolation_mode: str = "nearest",
    ) -> Field:
        """
        Propagates the field through free-space to a plane defined by the input parameters.

        Args:
            shape (Vector2): Number of grid points along the planar dimensions.
            z (Scalar): Position along the z-axis.
            spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default:
                if `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
            offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating to the plane.
        """
        return propagator(
            self, shape, z, spacing, offset, propagation_method, asm_pad_factor, interpolation_mode
        )

    def propagate_to_z(self, z: Scalar, **prop_kwargs) -> Field:
        """
        Propagates the field through free-space to a plane at a specific z position.

        The plane has the same ``shape``, ``spacing``, and ``offset`` as the input field.

        Args:
            z (Scalar): Position along the z-axis.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating to the plane.
        """

        return self.propagate(self.shape, z, self.spacing, self.offset, **prop_kwargs)

    def propagate_to_plane(self, plane: PlanarGrid, **prop_kwargs) -> Field:
        """
        Propagates the field through free-space to a plane defined by a :class:`PlanarGrid` object.

        Args:
            plane (PlanarGrid): Plane grid.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating to the plane.
        """

        if not isinstance(plane, PlanarGrid):
            raise TypeError(f"Expected plane to be a PlanarGrid, but got {type(plane).__name__}.")
        return self.propagate(plane.shape, plane.z, plane.spacing, plane.offset, **prop_kwargs)

    def modulate(self, modulation_profile: Tensor) -> Field:
        """
        Modulates the field by a modulation profile.

        Args:
            modulation_profile (Tensor): The modulation profile.

        Returns:
            Field: Modulated field.
        """
        modulated_data = self.data * modulation_profile
        return copy(self, data=modulated_data)

    def polarized_modulate(self, polarized_modulation_profile: Tensor) -> Field:
        """
        Modulates the field by a polarized modulation profile.

        Args:
            polarized_modulation_profile (Tensor): The polarized modulation profile.

        Returns:
            Field: Modulated field.
        """
        self._validate_polarization_dim()
        modulated_data = (self.data.unsqueeze(self.POLARIZATION_DIM - 1) * polarized_modulation_profile).sum(
            self.POLARIZATION_DIM
        )
        return copy(self, data=modulated_data)

    def polarized_split(self) -> tuple[Field, Field, Field]:
        """
        Splits the field into three polarized fields.

        Returns:
            tuple[Field, Field, Field]: The split fields.
        """
        self._validate_polarization_dim()
        fields = tuple(copy(self, data=torch.zeros_like(self.data)) for _ in range(3))
        for i in range(3):
            fields[i].data.select(self.POLARIZATION_DIM, i).copy_(self.data.select(self.POLARIZATION_DIM, i))
        return fields

    def normalize(self, normalized_power: Scalar = 1.0) -> Field:
        """
        Normalizes the field to a specified power.

        Args:
            normalized_power (Scalar): The normalized power. Default: `1.0`.

        Returns:
            Field: Normalized field.
        """
        ratio = torch.nan_to_num((normalized_power / self.power()[..., None, None]), 0)
        normalized_data = self.data * ratio.sqrt()

        return copy(self, data=normalized_data)

    def inner(self, other: Field) -> Tensor:
        """
        Returns the inner product of the field (last two data dimensions) with another field.

        Args:
            other (Field): The other field.

        Returns:
            Tensor: The inner product.
        """
        if not self.is_same_geometry(other):
            raise ValueError(
                "Fields must have the same geometry, but got geometries:"
                f"\n{self.geometry_str()}\n{other.geometry_str()}"
            )
        return inner2d(self.data, other.data) * self.cell_area()

    def outer(self, other: Field) -> Tensor:
        """
        Returns the outer product of the field (last two data dimensions) with another field.

        Args:
            other (Field): The other field.

        Returns:
            Tensor: The outer product.
        """
        if not self.is_same_geometry(other):
            raise ValueError(
                "Fields must have the same geometry, but got geometries:"
                f"\n{self.geometry_str()}\n{other.geometry_str()}"
            )
        return outer2d(self.data, other.data) * self.cell_area()

    def visualize(self, *index: int, **kwargs) -> Any:
        """
        Visualizes the field.

        Args:
            *index (int): Index of the data tensor to visualize.
            intensity (bool): Whether to visualize only the intensity. Default: `False`.
            **kwargs: Additional keyword arguments for visualization.
        """
        kwargs.update({"symbol": r"$\psi$"})
        return self._visualize(self.data, index, **kwargs)

    def _validate_polarization_dim(self) -> None:
        if self.data.ndim < abs(self.POLARIZATION_DIM) or self.data.shape[self.POLARIZATION_DIM] != 3:
            raise ValueError(
                f"Expected data tensor to have polarization dimension of size 3 at "
                f"dim={self.POLARIZATION_DIM}, but data has shape {self.data.shape}."
            )


class SpatialCoherence(Field):  # pylint: disable=abstract-method
    """
    Spatial Coherence class.

    Args:
        data (Tensor): The complex-valued spatial coherence data.
        wavelength (Scalar, optional): The wavelength of the field. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    DATA_MIN_NDIM = 4
    POLARIZATION_DIM = -5
    propagate = get_coherence_evolution(Field.propagate)
    modulate = get_coherence_evolution(Field.modulate)

    def intensity(self) -> Tensor:
        if self.data.shape[-1] != self.data.shape[-3] or self.data.shape[-2] != self.data.shape[-4]:
            shape_str = ", ".join(str(dim) for dim in self.data.shape)
            raise ValueError(f"Expected data tensor to have shape (..., H, W, H, W), but got ({shape_str}).")

        data_flattened = self.data.flatten(-4, -3).flatten(-2, -1)
        intensity = torch.diagonal(data_flattened, dim1=-2, dim2=-1).unflatten(-1, self.shape)
        if not torch.allclose(intensity.imag, torch.zeros_like(intensity.imag), atol=1e-7):
            raise ValueError(
                "Spatial coherence diagonal values are expected to be real, but significant imaginary "
                "components were found.\n"
                f"Max absolute real part: {intensity.real.abs().max().item():.4e}\n"
                f"Max absolute imaginary part: {intensity.imag.abs().max().item():.4e}\n"
            )

        return intensity.real

    def normalize(self, normalized_power: Scalar = 1.0) -> Field:
        ratio = torch.nan_to_num((normalized_power / self.power()[..., None, None, None, None]), 0)
        normalized_data = self.data * ratio
        return copy(self, data=normalized_data)

    def inner(self, other: Field) -> Tensor:
        """SpatialCoherence does not support the inner product."""
        raise TypeError("inner() is not applicable for SpatialCoherence.")

    def outer(self, other: Field) -> Tensor:
        """SpatialCoherence does not support the outer product."""
        raise TypeError("outer() is not applicable for SpatialCoherence.")

    def visualize(self, *index: int, **kwargs) -> Any:
        """
        Visualizes the the time-averaged intensity (diagonal of the spatial coherence matrix).

        Args:
            *index (int): Index of the data tensor to visualize.
            intensity (bool): Whether to visualize only the intensity. Default: `False`.
            **kwargs: Additional keyword arguments for visualization.
        """
        kwargs.update({"symbol": r"diag$(\Gamma)$"})
        return self._visualize(self.intensity(), index, **kwargs)
