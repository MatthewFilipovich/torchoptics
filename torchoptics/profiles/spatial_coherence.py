"""This module defines functions to generate spatial coherence function profiles."""

from typing import Callable, Optional

import torch
from torch import Tensor

from ..functional import outer2d
from ..planar_geometry import PlanarGeometry
from ..type_defs import Scalar, Vector2
from ..utils import initialize_tensor


def schell_model(
    shape: Vector2,
    intensity_func: Callable[[Tensor, Tensor], Tensor],
    coherence_func: Callable[[Tensor, Tensor], Tensor],
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
) -> Tensor:
    r"""
    Generates a spatial coherence function profile based on the Schell-model.

    The Schell model describes partially coherent light as a combination of an intensity distribution
    and a spatial coherence function. The mutual coherence function :math:`\Gamma(x_1, y_1, x_2, y_2)`
    is defined as:

    .. math::

        \Gamma(x_1, y_1, x_2, y_2) = \sqrt{I(x_1, y_1) \cdot I(x_2, y_2)} \cdot \mu(x_1 - x_2, y_1 - y_2),

    where:
        - :math:`I(x, y)` is the intensity distribution function, and
        - :math:`\mu(x_1 - x_2, y_1 - y_2)` is the spatial coherence function.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        intensity_func (Callable[[Tensor, Tensor], Tensor]): Function defining the intensity distribution,
            which takes the :math:`x` and :math:`y` coordinates and returns the intensity values.
        coherence_func (Callable[[Tensor, Tensor], Tensor]): Function defining the coherence distribution,
            which takes the pairwise :math:`dx` and :math:`dy` coordinate differences and returns the
            coherence values.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Offset coordinates of the pattern. Default: `(0, 0)`.

    Returns:
        torch.Tensor: A 4D tensor representing the mutual coherence function
        :math:`\Gamma(x_1, y_1, x_2, y_2)`.
    """
    x, y = PlanarGeometry(shape, spacing=spacing, offset=offset).meshgrid()
    intensity = intensity_func(x, y)

    # Compute pairwise differences for coherence function
    dx = x.unsqueeze(-1).unsqueeze(-1) - x.unsqueeze(0).unsqueeze(0)
    dy = y.unsqueeze(-1).unsqueeze(-1) - y.unsqueeze(0).unsqueeze(0)
    coherence = coherence_func(dx, dy)

    return outer2d(intensity, intensity) ** 0.5 * coherence


def gaussian_schell_model(
    shape: Vector2,
    waist_radius: Scalar,
    coherence_width: Scalar,
    spacing: Optional[Vector2] = None,
    offset: Optional[Vector2] = None,
) -> Tensor:
    r"""
    Generates a spatial coherence function profile based on the Gaussian Schell-model.

    The Gaussian Schell-model assumes both the intensity and coherence functions have Gaussian profiles.
    The mutual coherence function :math:`\Gamma(x_1, y_1, x_2, y_2)` is defined as:

    .. math::

        \Gamma(x_1, y_1, x_2, y_2) = \sqrt{I(x_1, y_1) \cdot I(x_2, y_2)} \cdot \mu(x_1 - x_2, y_1 - y_2)

    where:
        - :math:`I(x, y) = \exp{\left(-\frac{2(x^2 + y^2)}{w_0^2}\right)}` is the Gaussian intensity
          distribution with waist radius :math:`w_0`, and
        - :math:`\mu(x_1 - x_2, y_1 - y_2) = \exp{\left(-\frac{(x_1 - x_2)^2
          + (y_1 - y_2)^2}{2 \sigma_c^2}\right)}` is the Gaussian coherence function with coherence width
          :math:`\sigma_c`.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        waist_radius (Scalar): The beam waist radius.
        coherence_width (Scalar): The coherence width.
        spacing (Optional[Vector2]): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Optional[Vector2]): Offset coordinates of the pattern. Default: `(0, 0)`.

    Returns:
        torch.Tensor: A 4D tensor representing the mutual coherence function
        :math:`\Gamma(x_1, x_2, y_1, y_2)`.
    """

    def coherence_func(dx, dy):
        if coherence_width == 0:  # Return 1 only at (x, y) = (0, 0), and 0 elsewhere
            return (dx == 0) * (dy == 0)
        return torch.exp(-(dx**2 + dy**2) / (2 * coherence_width**2))

    def intensity_func(x, y):
        return 2 / (torch.pi * waist_radius**2) * torch.exp(-(2 * (x**2 + y**2)) / waist_radius**2)

    # Use the general Schell model to compute the coherence profile
    waist_radius = initialize_tensor("waist_radius", waist_radius, is_positive=True)
    return schell_model(shape, intensity_func, coherence_func, spacing, offset)
