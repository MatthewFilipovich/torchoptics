"""TorchOptics: Differentiable wave optics simulations with PyTorch."""

import torchoptics.elements as elements
import torchoptics.functional as functional
import torchoptics.profiles as profiles
import torchoptics.propagation as propagation
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
    # Configuration utilities
    "get_default_spacing",
    "get_default_wavelength",
    "set_default_spacing",
    "set_default_wavelength",
    # Visualization tools
    "animate_tensor",
    "visualize_tensor",
    # Submodules
    "elements",
    "functional",
    "profiles",
    "propagation",
]
