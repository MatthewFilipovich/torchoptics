"""
Custom Profiles
===============

Demonstrates how to create and combine custom optical profiles using TorchOptics.

We begin with simple geometric shapes, then construct a single lens profile,
and finally build a 3×3 lens array. These profiles are useful for simulating
apertures, phase masks, and other structured optical elements.
"""
# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = -1
# sphinx_gallery_end_ignore

import torch

from torchoptics import visualize_tensor
from torchoptics.profiles import circle, lens_phase, square, triangle

shape = 800

# %%
# Geometric Shape Profiles
# ------------------------
# First, we create a circular aperture profile centered on the grid.
# This can serve as a basic mask or aperture.
circle_profile = circle(shape, radius=50, spacing=1)
visualize_tensor(circle_profile, title="Circle Profile")

# %%
# Next, we create a square profile with an offset from the center of the grid.
# The offset is specified in physical units as (row, column), enabling precise
# placement of shapes when constructing composite masks.
#
# Note: When visualized with matplotlib, the [0, 0] index corresponds to the
# top-left corner of the grid. As a result, positive row offsets shift the shape
# downward, and positive column offsets shift it to the right.
square_profile = square(shape, side=100, offset=(0, 150), spacing=1)
visualize_tensor(square_profile, title="Square Profile (Offset)")

# %%
# Now we define a triangle profile with a specific base, height, and offset.
triangle_profile = triangle(shape, base=200, height=100, offset=(-150, 100), spacing=1)
visualize_tensor(triangle_profile, title="Triangle Profile (Offset)")

# %%
# We combine the circle, square, and triangle into a single profile.
# This illustrates how to compose more complex masks from simple components.
combined_profile = circle_profile + square_profile + triangle_profile
visualize_tensor(combined_profile, title="Combined Profile")

# %%
# Single Lens Profile
# -------------------
# Now let's create the phase profile of a thin lens and combine it with a circular aperture.
focal_length = 0.2  # Focal length in meters
wavelength = 800e-9  # Wavelength in meters
spacing = 10e-6  # Pixel spacing in meters
radius = 1e-3  # Lens radius in meters

# Generate the lens phase profile
phase = lens_phase(shape, focal_length, wavelength, spacing)
visualize_tensor(torch.exp(1j * phase), title="Lens Phase (exp(i·phase))")

# %%
# Create a circular aperture for the lens
circular_aperture = circle(shape, radius, spacing)
visualize_tensor(circular_aperture, title="Lens Circular Aperture")

# %%
# Multiply the aperture and phase to obtain the full lens profile
single_lens_profile = circular_aperture * torch.exp(1j * phase)
visualize_tensor(single_lens_profile, title="Single Lens Profile")

# %%
# Lens Array
# ----------
# To generate an array of lenses, we repeat and offset the single lens profile.
# Here, we create a 3×3 grid of identical lenses.
lens_array_profile = torch.zeros((shape, shape), dtype=torch.complex64)
for offset_i in [-2e-3, 0, 2e-3]:
    for offset_j in [-2e-3, 0, 2e-3]:
        offset = (offset_i, offset_j)
        phase = lens_phase(shape, focal_length, wavelength, spacing, offset)
        circular_aperture = circle(shape, radius, spacing, offset)
        lens_array_profile += circular_aperture * torch.exp(1j * phase)

visualize_tensor(lens_array_profile, title="Lens Array Profile (3×3)")
