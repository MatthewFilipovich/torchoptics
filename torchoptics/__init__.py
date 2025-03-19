"""TorchOptics is an open-source Python library for differentiable wave optics simulations with PyTorch."""

import torchoptics.elements
import torchoptics.functional
import torchoptics.profiles
from torchoptics.config import (
    get_default_spacing,
    get_default_wavelength,
    set_default_spacing,
    set_default_wavelength,
)
from torchoptics.fields import CoherenceField, Field, PolarizedField
from torchoptics.optics_module import OpticsModule
from torchoptics.planar_geometry import PlanarGeometry
from torchoptics.system import System
from torchoptics.visualization import visualize_tensor
