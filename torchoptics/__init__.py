"""TorchOptics: Differentiable wave optics simulations with PyTorch."""

import torchoptics.elements
import torchoptics.functional
import torchoptics.profiles
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
