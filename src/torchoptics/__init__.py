"""TorchOptics: Differentiable wave optics simulation library built on PyTorch."""

from torchoptics import elements, functional, profiles, propagation
from torchoptics.config import (
    get_default_dtype,
    get_default_spacing,
    get_default_wavelength,
    set_default_dtype,
    set_default_spacing,
    set_default_wavelength,
)
from torchoptics.fields import Field, SpatialCoherence
from torchoptics.optics_module import OpticsModule
from torchoptics.planar_grid import PlanarGrid
from torchoptics.system import System
from torchoptics.visualization import animate_tensor, visualize_tensor

from ._version import __version__

__all__ = [
    "Field",
    "OpticsModule",
    "PlanarGrid",
    "SpatialCoherence",
    "System",
    "animate_tensor",
    "elements",
    "get_default_dtype",
    "functional",
    "get_default_spacing",
    "get_default_wavelength",
    "profiles",
    "propagation",
    "set_default_dtype",
    "set_default_spacing",
    "set_default_wavelength",
    "visualize_tensor",
    "__version__",
]
