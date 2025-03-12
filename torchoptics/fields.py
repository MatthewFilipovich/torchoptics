"""This module defines the Field, PolarizedField, and CoherenceField classes."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from .config import wavelength_or_default
from .functional import calculate_centroid, calculate_std, inner2d, outer2d
from .planar_geometry import PlanarGeometry
from .propagation import propagator
from .type_defs import Scalar, Vector2
from .utils import copy

__all__ = ["Field", "PolarizedField", "CoherenceField"]


class Field(PlanarGeometry):  # pylint: disable=abstract-method
    """
    Scalar optical field.

    Args:
        data (Tensor): The complex-valued field data.
        wavelength (Scalar, optional): The wavelength of the field. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    _data_min_dim = 2
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

        self._validate_data(data)
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
            field (Field): Input field.
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
            field (Field): Input field.
            z (Scalar): Position along the z-axis.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating to the plane.
        """

        return self.propagate(self.shape, z, self.spacing, self.offset, **prop_kwargs)

    def propagate_to_plane(self, plane: PlanarGeometry, **prop_kwargs) -> Field:
        """
        Propagates the field through free-space to a plane defined by a :class:`PlanarGeometry` object.

        Args:
            field (Field): Input field.
            plane (PlanarGeometry): Plane geometry.
            propagation_method (str): The propagation method to use. Default: `"AUTO"`.
            asm_pad_factor (Vector2): The padding factor along both planar dimensions for ASM propagation.
                Default: `2`.
            interpolation_mode (str): The interpolation mode to use. Default: `"nearest"`.


        Returns:
            Field: Output field after propagating to the plane.
        """

        if not isinstance(plane, PlanarGeometry):
            raise TypeError(f"Expected plane to be a PlanarGeometry, but got {type(plane).__name__}.")
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

    def normalize(self, normalized_power: Scalar = 1.0) -> Field:
        """
        Normalizes the field to a specified power.

        Args:
            normalized_power (Scalar): The normalized power. Default: `1.0`.

        Returns:
            Field: Normalized field.
        """
        indices_unsqueezed = [..., *((None,) * self._data_min_dim)]
        normalized_data = self.data * (normalized_power / self.power()[indices_unsqueezed]).sqrt()
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

    def _validate_data(self, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Expected data to be a tensor, but got {type(tensor).__name__}.")
        if tensor.dim() < self._data_min_dim:
            raise ValueError(
                f"Expected data to have at least {self._data_min_dim} dimensions, but got {tensor.dim()}."
            )


class PolarizedField(Field):  # pylint: disable=abstract-method
    """
    Polarized optical field.

    Args:
        data (Tensor): The complex-valued polarized field data.
        wavelength (Scalar, optional): The wavelength of the field. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    _data_min_dim = 3

    def intensity(self) -> Tensor:
        return super().intensity().sum(dim=-3)  # Sum over the polarization dimension.

    def polarized_modulate(self, polarized_modulation_profile: Tensor) -> PolarizedField:
        """
        Modulates the field by a polarized modulation profile.

        Args:
            polarized_modulation_profile (Tensor): The polarized modulation profile.

        Returns:
            Field: Modulated field.
        """
        modulated_data = (self.data.unsqueeze(-4) * polarized_modulation_profile).sum(-3)
        return copy(self, data=modulated_data)  # type: ignore[return-value]

    def polarized_split(self) -> tuple[PolarizedField, PolarizedField, PolarizedField]:
        """
        Splits the field into three polarized fields.

        Returns:
            tuple[Field, Field, Field]: The split fields.
        """
        fields = tuple(copy(self, data=torch.zeros_like(self.data)) for _ in range(3))
        for i in range(3):
            fields[i].data.select(-3, i).copy_(self.data.select(-3, i))
        return fields  # type: ignore[return-value]

    def _validate_data(self, tensor: Tensor) -> None:
        super()._validate_data(tensor)
        if tensor.shape[-3] != 3:
            raise ValueError(
                f"Expected data to have a size of 3 in the polarization dimension (-3), "
                f"but got {tensor.shape[-3]}."
            )


class CoherenceField(Field):  # pylint: disable=abstract-method
    """
    Scalar optical spatial coherence.

    Args:
        data (Tensor): The complex-valued spatial coherence data.
        wavelength (Scalar, optional): The wavelength of the field. Default: if `None`, uses a global default
            (see :meth:`torchoptics.set_default_wavelength()`).
        z (Scalar): Position along the z-axis. Default: `0`.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    _data_min_dim = 4

    def intensity(self) -> Tensor:
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
        def adjoint(data: Tensor):
            return data.conj().transpose(-1, -3).transpose(-2, -4)

        def prop(data: Tensor, output_plane_geometry: dict):
            return propagator(
                copy(self, data=data),
                **output_plane_geometry,
                propagation_method=propagation_method,
                asm_pad_factor=asm_pad_factor,
                interpolation_mode=interpolation_mode,
            ).data

        # Define the geometry of the output plane for propagation.
        output_geometry = PlanarGeometry(shape, z, spacing, offset).geometry
        propagated_data = adjoint(prop(adjoint(prop(self.data, output_geometry)), output_geometry))
        return copy(self, data=propagated_data, z=z, spacing=spacing, offset=offset)

    def modulate(self, modulation_profile: Tensor) -> Field:
        modulated_data = self.data * outer2d(modulation_profile, modulation_profile)
        return copy(self, data=modulated_data)

    def normalize(self, normalized_power: Scalar = 1.0) -> Field:
        indices_unsqueezed = [..., *((None,) * self._data_min_dim)]
        normalized_data = self.data * (normalized_power / self.power()[indices_unsqueezed])
        return copy(self, data=normalized_data)

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
