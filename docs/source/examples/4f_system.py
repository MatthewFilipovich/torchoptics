"""
4f System
==========

Simulates 4f optical systems with two lenses using low-pass and high-pass filters.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens, Modulator
from torchoptics.profiles import checkerboard, circle

# %%
# Simulation Parameters
# ---------------------
# We define the grid size, spacing, wavelength, and optical system properties.

shape = 1000  # Grid size (number of points per dimension)
spacing = 10e-6  # Grid spacing (m)
wavelength = 700e-9  # Wavelength (m)
focal_length = 200e-3  # Lens focal length (m)

# Checkerboard parameters for the input field
tile_length = 400e-6  # Tile size (m)
num_tiles = 15  # Number of tiles in each dimension

# Select computation device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure torchoptics defaults
torchoptics.set_default_spacing(spacing)
torchoptics.set_default_wavelength(wavelength)

# %%
# Input Field: Checkerboard Pattern
# ---------------------------------
# We generate a checkerboard pattern as the input field.

field_data = checkerboard(shape, tile_length, num_tiles)
input_field = Field(field_data).to(device)
input_field.visualize(title="Input Field")

# %%
# 4f Optical System
# -----------------
# We define a basic 4f system with two lenses.

system = System(
    Lens(shape, focal_length, z=1 * focal_length),
    Lens(shape, focal_length, z=3 * focal_length),
).to(device)

# Visualize the lenses
system[0].visualize(title="Lens 1")
system[1].visualize(title="Lens 2")

# %%
# Measuring Field at Focal Planes
# --------------------------------
# We capture the field at various focal plane distances.

z_positions = [i * focal_length for i in range(5)]
measurements = [system.measure_at_z(input_field, z=z) for z in z_positions]

# Visualize intensity distributions
for i, measurement in enumerate(measurements):
    measurement.visualize(title=f"z = {i}f", vmax=1)

# %%
# Low-Pass Filter
# ---------------
# We apply a low-pass filter in the Fourier plane by inserting a circular aperture.

radius = 500e-6  # Low-pass filter radius (m)
low_pass_modulation_profile = circle(shape, radius)

low_pass_system = System(
    Lens(shape, focal_length, z=1 * focal_length),
    Modulator(low_pass_modulation_profile, z=2 * focal_length),
    Lens(shape, focal_length, z=3 * focal_length),
).to(device)

# Visualize the low-pass filter
low_pass_system[1].visualize(title="Low-Pass Filter")

# Measure and visualize results
measurements = [low_pass_system.measure_at_z(input_field, z=z) for z in z_positions]

for i, measurement in enumerate(measurements):
    measurement.visualize(title=f"z = {i}f", vmax=1)

# %%
# High-Pass Filter
# ----------------
# We apply a high-pass filter by taking the logical complement of the low-pass filter.

high_pass_modulation_profile = low_pass_modulation_profile.logical_not()

high_pass_system = System(
    Lens(shape, focal_length, z=1 * focal_length),
    Modulator(high_pass_modulation_profile, z=2 * focal_length),
    Lens(shape, focal_length, z=3 * focal_length),
).to(device)

# Visualize the high-pass filter
high_pass_system[1].visualize(title="High-Pass Filter")

# Measure and visualize results
measurements = [high_pass_system.measure_at_z(input_field, z=z) for z in z_positions]

for i, measurement in enumerate(measurements):
    measurement.visualize(title=f"z = {i}f", vmax=1)
