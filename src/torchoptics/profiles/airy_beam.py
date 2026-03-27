"""Airy beam profile generation."""

import torch
from torch import Tensor

from ..types import Scalar, Vector2
from ..utils import initialize_tensor
from ._profile_meshgrid import profile_meshgrid


def airy_beam(
    shape: Vector2,
    scale: Scalar,
    truncation: Scalar,
    spacing: Vector2 | None = None,
    offset: Vector2 | None = None,
) -> Tensor:
    r"""Generate a truncated 2D Airy beam profile.

    The profile is defined as the separable product of two 1D truncated Airy
    functions:

    .. math::
        \psi(x, y) = \operatorname{Ai}(x / x_0) e^{a x / x_0}
        \operatorname{Ai}(y / x_0) e^{a y / x_0}

    where:

    - :math:`\operatorname{Ai}` is the Airy function of the first kind,
    - :math:`x_0` is the transverse scale that sets the lobe spacing, and
    - :math:`a` is the exponential truncation factor that makes the beam finite
      energy.

    Args:
        shape (Vector2): Number of grid points along the planar dimensions.
        scale (Scalar): The transverse Airy scale.
        truncation (Scalar): The exponential truncation factor.
        spacing (Vector2 | None): Distance between grid points along planar dimensions. Default: if
            `None`, uses a global default (see :meth:`torchoptics.set_default_spacing()`).
        offset (Vector2 | None): Center coordinates of the profile. Default: `(0, 0)`.

    Returns:
        Tensor: The generated Airy beam profile.

    """
    scale = initialize_tensor("scale", scale, is_scalar=True, is_positive=True)
    truncation = initialize_tensor("truncation", truncation, is_scalar=True, is_non_negative=True)

    x, y = profile_meshgrid(shape, spacing, offset)
    scaled_x = x / scale
    scaled_y = y / scale

    airy_x = torch.special.airy_ai(scaled_x) * torch.exp(truncation * scaled_x)
    airy_y = torch.special.airy_ai(scaled_y) * torch.exp(truncation * scaled_y)
    return airy_x * airy_y
