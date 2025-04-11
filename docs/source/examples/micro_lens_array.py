"""
Microlens Array Imaging
=======================

Simulates a 5x5 microlens array and demonstrates its focusing behavior on an incoming Gaussian beam.
"""

# %%
import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Modulator, PhaseModulator
from torchoptics.fields import Field
from torchoptics.profiles import circle, gaussian, lens_phase

# %%
# Simulation Setup
# ----------------
# Set the default grid spacing and wavelength for the simulation.

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Define a Custom Microlens Array Element
# ---------------------------------------
# A microlens array is created by tiling multiple lens phase profiles across the grid.

shape = 500
focal_length = 50e-3  # Lens focal length (m)
aperture_radius = 0.5e-3  # 0.5 mm

microlens_phase = 0
for i in range(3):
    for j in range(3):
        offset = (i * 2 * aperture_radius, j * 2 * aperture_radius)
        microlens_phase += lens_phase(shape, focal_length, aperture_radius, offset=offset) * circle(
            shape, aperture_radius, offset=offset
        )

microlens = PhaseModulator(microlens_phase)
microlens.visualize()


# %%
# Generate Input Gaussian Beam
# ----------------------------
# The beam simulates an extended source illuminating the microlens array.

input_profile = gaussian((shape, shape), waist_radius=0.3e-3)
input_field = Field(input_profile, z=-10e-3)  # Input field at z = -10 mm

input_field.visualize(title="Input Gaussian Beam")

# %%
# Build and Apply Microlens Array
# -------------------------------
# The microlens array is placed at z = 0. Each lens focuses light to a shared focal plane.


# Build optical system
system = System(microlens)

# %%
# Observe the Focal Plane
# -----------------------
# We simulate image formation at the shared focal plane of the microlens array.

image_z = focal_length  # focus distance

field_image = system.measure_at_z(input_field, z=image_z)
field_image.visualize(title=f"Microlens Array Focus (z = {image_z} m)")
