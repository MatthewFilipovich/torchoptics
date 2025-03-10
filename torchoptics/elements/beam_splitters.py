"""This module defines the beam splitter element."""

from typing import Optional

import torch
from torch import Tensor

from ..fields import Field, PolarizedField
from ..type_defs import Scalar, Vector2
from ..utils import copy
from .elements import Element

__all__ = ["BeamSplitter", "PolarizingBeamSplitter"]


class BeamSplitter(Element):
    r"""
    Beam splitter element.

    The beam splitter is described by the following `transfer matrix 
    <https://en.wikipedia.org/wiki/Beam_splitter>`_:

    .. math::
        \tau = e^{i\phi_0}
        \begin{bmatrix}
            \sin \theta e^{i\phi_R} & \cos \theta e^{-i\phi_T} \\
            \cos \theta e^{i\phi_T} & -\sin \theta e^{-i\phi_R}
        \end{bmatrix}

    .. note::
        A 50:50 beam splitter is obtained by setting :math:`\theta = \pi/4`. 
        
        The dielectric 50:50 beam splitter has :math:`\phi_T = \phi_R = \phi_0 = 0`, while the symmetric 
        beam splitter of Loudon (also 50:50) has :math:`\phi_T = 0`, :math:`\phi_R = -\pi/2`, and 
        :math:`\phi_0 = \pi/2`. 

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        z (Scalar): Position along the z-axis. Default: `0`. 
        theta (Scalar): Angle relating the transmitted and reflected beams.
        phi_0 (Scalar): Global phase shift.
        phi_r (Scalar): Reflection phase shift.
        phi_t (Scalar): Transmission phase shift.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Center coordinates of the plane. Default: `(0, 0)`.
    """

    theta: Tensor
    phi_0: Tensor
    phi_r: Tensor
    phi_t: Tensor

    def __init__(
        self,
        shape: Vector2,
        theta: Scalar,
        phi_0: Scalar,
        phi_r: Scalar,
        phi_t: Scalar,
        z: Scalar = 0,
        spacing: Optional[Vector2] = None,
        offset: Optional[Vector2] = None,
    ) -> None:
        super().__init__(shape, z, spacing, offset)
        self.register_optics_property("theta", theta, is_scalar=True)
        self.register_optics_property("phi_0", phi_0, is_scalar=True)
        self.register_optics_property("phi_r", phi_r, is_scalar=True)
        self.register_optics_property("phi_t", phi_t, is_scalar=True)

    @property
    def transfer_matrix(self) -> Tensor:
        """Return the transfer matrix of the beam splitter."""
        transfer_matrix = torch.zeros(2, 2, dtype=torch.cdouble, device=self.theta.device)
        transfer_matrix[0, 0] = torch.sin(self.theta) * torch.exp(1j * self.phi_r)
        transfer_matrix[0, 1] = torch.cos(self.theta) * torch.exp(-1j * self.phi_t)
        transfer_matrix[1, 0] = torch.cos(self.theta) * torch.exp(1j * self.phi_t)
        transfer_matrix[1, 1] = -torch.sin(self.theta) * torch.exp(-1j * self.phi_r)
        return transfer_matrix

    def forward(self, field: Field, other: Optional[Field] = None) -> tuple[Field, Field]:
        """
        Applies the beam splitter to the input fields.

        Args:
            field (Field): The field to split.
            other (Optional[Field]): The other field to split. Default: `None`.

        Returns:
            tuple[Field, Field]: The split fields.
        """
        self.validate_field(field)
        output_data0 = field.data * self.transfer_matrix[0, 0]
        output_data1 = field.data * self.transfer_matrix[1, 0]
        if other:
            self.validate_field(other)
            output_data0 += other.data * self.transfer_matrix[0, 1]
            output_data1 += other.data * self.transfer_matrix[1, 1]

        return copy(field, data=output_data0), copy(field, data=output_data1)


class PolarizingBeamSplitter(Element):
    """
    Polarizing beam splitter element.

    The polarizing beam splitter splits the input field into two orthogonal polarizations.
    """

    def forward(self, field: PolarizedField) -> tuple[PolarizedField, PolarizedField]:
        """
        Applies the beam splitter to the input field.

        Args:
            field (PolarizedField): The field to split.

        Returns:
            tuple[PolarizedField, PolarizedField]: The split fields.
        """
        self.validate_field(field)
        if not torch.all(field.data.select(-3, 2) == 0):
            raise ValueError("Polarized field cannot have z polarization.")
        return field.polarized_split()[:2]
