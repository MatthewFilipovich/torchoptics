"""TorchOptics: Differentiable wave optics simulations with PyTorch."""

from torchoptics import elements, functional, profiles, propagation
from torchoptics.config import (
    get_default_spacing,
    get_default_wavelength,
    set_default_spacing,
    set_default_wavelength,
)
from torchoptics.fields import Field, SpatialCoherence
from torchoptics.optics_module import OpticsModule
from torchoptics.planar_grid import PlanarGrid
from torchoptics.system import System
from torchoptics.visualization import animate_tensor, visualize_tensor

__all__ = [
    # Core classes
    "Field",
    "OpticsModule",
    "PlanarGrid",
    "SpatialCoherence",
    "System",
    # Visualization tools
    "animate_tensor",
    # Submodules
    "elements",
    "functional",
    # Configuration utilities
    "get_default_spacing",
    "get_default_wavelength",
    "profiles",
    "propagation",
    "set_default_spacing",
    "set_default_wavelength",
    "visualize_tensor",
]
