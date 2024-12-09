"""This module defines the Field, PolarizedField, and CoherenceField classes."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from .config import get_default_wavelength
from .functional import calculate_centroid, calculate_std, initialize_tensor, inner2d, outer2d
from .param import Param
from .planar_geometry import PlanarGeometry
from .propagation import VALID_INTERPOLATION_MODES, VALID_PROPAGATION_METHODS, propagator
from .type_defs import Scalar, Vector2

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
        propagation_method (str): The propagation method to use. Default: `"AUTO"`.
        asm_pad_factor (Vector2): The padding factor along planar dimensions for angular spectrum method (ASM)
            propagation. Default: `2`.
        interpolation_mode (str): The interpolation mode to use. Default: `"bilinear"`.
    """

    _data_min_dim = 2

    def __init__(
        self,
        data: Tensor,
        wavelength: Optional[Scalar] = None,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
        propagation_method: str = "AUTO",
        asm_pad_factor: Vector2 = 2,
        interpolation_mode: str = "bilinear",
    ) -> None:

        self._validate_data(data)
        super().__init__(data.shape[-2:], z, spacing, offset)
        self.register_optics_property("data", data, is_complex=True)
        self.register_optics_property(
            "wavelength",
            get_default_wavelength() if wavelength is None else wavelength,
            (),
            validate_positive=True,
        )

        self.propagation_method = propagation_method
        self.asm_pad_factor = asm_pad_factor  # type: ignore
        self.interpolation_mode = interpolation_mode

    @property
    def propagation_method(self) -> str:
        """Returns the propagation method."""
        return self._propagation_method

    @propagation_method.setter
    def propagation_method(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Expected propagation_method to be a string, but got {type(value).__name__}.")
        if value.upper() not in VALID_PROPAGATION_METHODS:
            raise ValueError(
                f"Expected propagation_method to be one of {VALID_PROPAGATION_METHODS}, but got {value}."
            )
        self._propagation_method = value.upper()

    @property
    def asm_pad_factor(self) -> tuple:
        """Returns the padding factor for angular spectrum method (ASM) propagation."""
        return self._asm_pad_factor

    @asm_pad_factor.setter
    def asm_pad_factor(self, value: Vector2) -> None:
        tensor = initialize_tensor(
            "asm_pad_factor", value, (2,), is_integer=True, validate_non_negative=True, fill_value=True
        )
        self._asm_pad_factor = (tensor[0].item(), tensor[1].item())

    @property
    def interpolation_mode(self) -> str:
        """Returns the interpolation mode."""
        return self._interpolation_mode

    @interpolation_mode.setter
    def interpolation_mode(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Expected interpolation_mode to be a string, but got {type(value).__name__}.")
        if value.lower() not in VALID_INTERPOLATION_MODES:
            raise ValueError(
                f"Expected interpolation_mode to be one of {VALID_INTERPOLATION_MODES}, but got {value}."
            )
        self._interpolation_mode = value.lower()

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", propagation_method={self.propagation_method}"
            + f", asm_pad_factor=({self.asm_pad_factor[0]}, {self.asm_pad_factor[1]})"
            + f", interpolation_mode={self.interpolation_mode})"
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

        Returns:
            Field: Output field after propagating to the plane.
        """
        return propagator(self, shape, z, spacing, offset)

    def propagate_to_z(self, z: Scalar) -> Field:
        """
        Propagates the field through free-space to a plane at a specific z position.

        The plane has the same ``shape``, ``spacing``, and ``offset`` as the input field.

        Args:
            field (Field): Input field.
            z (Scalar): Position along the z-axis.

        Returns:
            Field: Output field after propagating to the plane.
        """

        return self.propagate(self.shape, z, self.spacing, self.offset)

    def propagate_to_plane(self, plane: PlanarGeometry) -> Field:
        """
        Propagates the field through free-space to a plane defined by a :class:`PlanarGeometry` object.

        Args:
            field (Field): Input field.
            plane (PlanarGeometry): Plane geometry.

        Returns:
            Field: Output field after propagating to the plane.
        """

        if not isinstance(plane, PlanarGeometry):
            raise TypeError(f"Expected plane to be a PlanarGeometry, but got {type(plane).__name__}.")
        return self.propagate(plane.shape, plane.z, plane.spacing, plane.offset)

    def modulate(self, modulation_profile: Tensor) -> Field:
        """
        Modulates the field by a modulation profile.

        Args:
            modulation_profile (Tensor): The modulation profile.

        Returns:
            Field: Modulated field.
        """
        modulated_data = self.data * modulation_profile
        return self.copy(data=modulated_data)

    def normalize(self, normalized_power: float = 1.0) -> Field:
        """
        Normalizes the field to a specified power.

        Args:
            normalized_power (float): The normalized power. Default: `1.0`.

        Returns:
            Field: Normalized field.
        """
        indices_unsqueezed = [..., *((None,) * self._data_min_dim)]
        normalized_data = self.data * (normalized_power / self.power()[indices_unsqueezed]).sqrt()
        return self.copy(data=normalized_data)

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
        kwargs.update({"symbol": r"\psi"})
        return self._visualize(self.data, index, **kwargs)

    def copy(self, **kwargs) -> Field:
        """
        Copies the field with updated properties.

        Args:
            **kwargs: New properties to update.

        Returns:
            Field: Copied field.
        """
        properties = {
            "data": self.data,
            "wavelength": self.wavelength,
            "z": self.z,
            "spacing": self.spacing,
            "offset": self.offset,
            "propagation_method": self.propagation_method,
            "asm_pad_factor": self.asm_pad_factor,
            "interpolation_mode": self.interpolation_mode,
        }
        properties.update(kwargs)
        return self.__class__(**properties)

    def _validate_data(self, tensor: Tensor) -> None:
        if not isinstance(tensor, (Tensor, Param)):
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
        propagation_method (str): The propagation method to use. Default: `"AUTO"`.
        asm_pad_factor (Vector2): The padding factor along planar dimensions for angular spectrum method (ASM)
            propagation. Default: `2`.
        interpolation_mode (str): The interpolation mode to use. Default: `"bilinear"`.
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
        return self.copy(data=modulated_data)  # type: ignore

    def polarized_split(self) -> tuple[PolarizedField, PolarizedField, PolarizedField]:
        """
        Splits the field into three polarized fields.

        Returns:
            tuple[Field, Field, Field]: The split fields.
        """
        fields = tuple(self.copy(data=torch.zeros_like(self.data)) for _ in range(3))
        for i in range(3):
            fields[i].data.select(-3, i).copy_(self.data.select(-3, i))
        return fields  # type: ignore

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
        propagation_method (str): The propagation method to use. Default: `"AUTO"`.
        asm_pad_factor (Vector2): The padding factor along planar dimensions for angular spectrum method (ASM)
            propagation. Default: `2`.
        interpolation_mode (str): The interpolation mode to use. Default: `"bilinear"`.
    """

    _data_min_dim = 4

    def intensity(self) -> Tensor:
        data_flattened = self.data.flatten(-4, -3).flatten(-2, -1)
        intensity = torch.diagonal(data_flattened, dim1=-2, dim2=-1).unflatten(-1, self.shape)
        if not torch.allclose(intensity.imag, torch.zeros_like(intensity.imag)):
            raise ValueError("The diagonal values of the spatial coherence are not all real.")
        return intensity.real

    def propagate(
        self,
        shape: Vector2,
        z: Scalar,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> Field:
        def adjoint(data: Tensor):
            return data.conj().transpose(-1, -3).transpose(-2, -4)

        def prop(data: Tensor, output_plane_geometry: dict):
            return propagator(self.copy(data=data), **output_plane_geometry).data

        # Define the geometry of the output plane for propagation.
        output_geometry = PlanarGeometry(shape, z, spacing, offset).geometry
        propagated_data = adjoint(prop(adjoint(prop(self.data, output_geometry)), output_geometry))
        return self.copy(data=propagated_data, z=z, spacing=spacing, offset=offset)

    def modulate(self, modulation_profile: Tensor) -> Field:
        modulated_data = self.data * outer2d(modulation_profile, modulation_profile)
        return self.copy(data=modulated_data)

    def normalize(self, normalized_power: Scalar = 1.0) -> Field:
        indices_unsqueezed = [..., *((None,) * self._data_min_dim)]
        normalized_data = self.data * (normalized_power / self.power()[indices_unsqueezed])
        return self.copy(data=normalized_data)
