"""
Animations
==========

Demonstrates how to create animations of field propagation in TorchOptics.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field, animate_tensor
from torchoptics.profiles import circle

# Set default simulation parameters
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# Initialize a 1D field
field = Field(circle(200, radius=50e-5))
# field.visualize(vmax=2)

# %%
# Propagate

z_values = torch.linspace(0, 0.5, 101)
intensities = torch.stack([field.propagate_to_z(z).intensity() for z in z_values])

# %%
# Animate
animate_tensor(intensities, vmax=2)
