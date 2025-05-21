"""Diffraction Pattern.
===================

Computes the diffraction pattern from an array of circular apertures.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field, visualize_tensor
from torchoptics.profiles import circle

# %%
# Simulation Parameters
# ---------------------
# Define the grid size, aperture properties, and propagation settings.

circle_shape = (800, 100)  # Grid size (y, x)
circle_radius = 200e-6  # Radius of each aperture (m)
circle_separation = 2e-3  # Separation between apertures (m)

# Define propagation properties
z_max = 2  # Maximum propagation distance (m)
z_num = 1000  # Number of propagation steps
propagation_shape = (800, 1)  # Grid size during propagation (y, x)

# Select computation device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure torchoptics defaults
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Input Field: Circular Aperture Array
# ------------------------------------
# We define an input field consisting of three circular apertures.

input_field = Field(
    circle(circle_shape, circle_radius, offset=(-circle_separation, 0))
    + circle(circle_shape, circle_radius, offset=(0, 0))
    + circle(circle_shape, circle_radius, offset=(circle_separation, 0)),
).to(device)

input_field.visualize(title="Input Aperture Field")

# %%
# Field Propagation and Diffraction Pattern
# -----------------------------------------
# We propagate the input field over a range of distances and store the intensity.

propagation_distances = torch.linspace(0, z_max, z_num)  # Propagation distances (m)
intensities = torch.zeros((z_num, propagation_shape[0]), device=device)  # Store intensities at each step

# Propagate the field and record intensity
for i, z in enumerate(propagation_distances):
    output_field = input_field.propagate(shape=propagation_shape, z=z)
    intensities[i] = output_field.intensity().flatten()

# %%
# Visualizing Diffraction Pattern Over Distance
# ---------------------------------------------
# We visualize the intensity distribution as a function of propagation distance.

visualize_tensor(intensities.T, xlabel="Propagation Distance (z)", ylabel="y", vmax=0.2)

# %%
# Intensity Profile at Maximum Distance
# -------------------------------------
# Finally, we plot the intensity profile at the maximum propagation distance.

plt.plot(intensities[-1].cpu())
plt.xlabel("y")
plt.ylabel("Intensity")
plt.xlim(0, propagation_shape[0])
plt.ylim(0)
plt.title("Intensity Profile at z_max")
plt.show()
