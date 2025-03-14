"""
4f System
==========

This example demonstrates the use of a 4f optical system to measure the intensity distribution of a field.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore
import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens
from torchoptics.profiles import checkerboard

# %%
# Set simulation properties
shape = 1000  # Number of grid points in each dimension
spacing = 10e-6  # Spacing between grid points (m)
wavelength = 700e-9  # Field wavelength (m)
focal_length = 200e-3  # Lens focal length (m)
tile_length = 400e-6  # Checkerboard tile length (m)
num_tiles = 15  # Number of tiles in each dimension

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure torchoptics default properties
torchoptics.set_default_spacing(spacing)
torchoptics.set_default_wavelength(wavelength)

# %%
# Initialize input field with checkerboard pattern
field_data = checkerboard(shape, tile_length, num_tiles)
input_field = Field(field_data).to(device)

input_field.visualize(title="Input field")

# %%
# Define 4f optical system with two lenses
system = System(
    Lens(shape, focal_length, z=1 * focal_length),
    Lens(shape, focal_length, z=3 * focal_length),
).to(device)

system[0].visualize(title="Lens 1")
system[1].visualize(title="Lens 2")

# %%
# Measure field at focal planes along the z-axis
measurements = [system.measure_at_z(input_field, z=i * focal_length) for i in range(5)]

# %%
# Visualize the measured intensity distributions
for i, measurement in enumerate(measurements):
    measurement.visualize(title=f"z={i}f", vmax=1)

# %% [markdown]
# ## Low-pass filter

# %%
from torchoptics.elements import Modulator
from torchoptics.profiles import circle

# %%
radius = 500e-6

low_pass_modulation_profile = circle(shape, radius)
low_pass_system = System(
    Lens(shape, focal_length, z=1 * focal_length),
    Modulator(low_pass_modulation_profile, z=2 * focal_length),
    Lens(shape, focal_length, z=3 * focal_length),
).to(device)

# Visualize the low-pass filter
low_pass_system[1].visualize(title="Low-pass filter")

# %%
# Measure field at different positions along the z-axis
measurements = [low_pass_system.measure_at_z(input_field, z=i * focal_length) for i in range(5)]

# %%
# Visualize the measured fields
for i, measurement in enumerate(measurements):
    measurement.visualize(title=f"z={i}f", vmax=1)

# %% [markdown]
# ## High-pass filter
#

# %%
high_pass_modulation_profile = low_pass_modulation_profile.logical_not()
high_pass_system = System(
    Lens(shape=1000, focal_length=focal_length, z=1 * focal_length),
    Modulator(high_pass_modulation_profile, z=2 * focal_length),
    Lens(shape=1000, focal_length=focal_length, z=3 * focal_length),
).to(device)

# Visualize the high-pass filter
high_pass_system[1].visualize(title="High-pass filter")

# %%
# Measure field at different positions along the z-axis
measurements = [high_pass_system.measure_at_z(input_field, z=i * focal_length) for i in range(5)]

# %%
# Visualize the measured fields
for i, measurement in enumerate(measurements):
    measurement.visualize(title=f"z={i}f", vmax=1)
