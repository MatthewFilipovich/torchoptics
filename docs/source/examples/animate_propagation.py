"""
Animate Propagation
===================

Demonstrates how to create animations of field propagation in TorchOptics.
"""

import torch

import torchoptics
from torchoptics import Field
from torchoptics.profiles import circle
from torchoptics.visualization import animate_tensor

# Set default simulation parameters
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# Initialize field with a circular profile
field = Field(circle(200, radius=50e-5))
field.visualize(vmax=2)

# %%
# Propagate the field

z_values = torch.linspace(0, 0.5, 201)
intensities = torch.stack([field.propagate_to_z(z).intensity() for z in z_values])

# %%
# Animate

vmax = 3
titles = [f"z = {z:.2f} m" for z in z_values]
func_anim_kwargs = {"interval": 100}

animate_tensor(intensities, vmax=vmax, title=titles, func_anim_kwargs=func_anim_kwargs)
