"""
Custom Profiles
=================

Demonstrates how to create custom profiles such as lens arrays.
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
# Profile with shapes
# -------------------
# Let's create a custom 2D profile containing a circle, square, and triangle.
# We will visualize these profiles using the `visualize_tensor` function.
circle_profile = circle(shape, radius=50, spacing=1)
visualize_tensor(circle_profile)

# %%
# Next we will create a square with an offset from the center of the grid.
# Note: the offset uses (i, j) indexing, where i is the row and j is the column.
# By default, matplotlib places the origin at the top-left corner.
square_profile = square(shape, side=100, offset=(0, 150), spacing=1)
visualize_tensor(square_profile)
# %%
# Finally, we will create a triangle profile with a specified base and height.
triangle_profile = triangle(shape, base=200, height=100, offset=(-150, 100), spacing=1)
visualize_tensor(triangle_profile)

# %%
# Now, let's combine these profiles into a single profile.
combined_profile = circle_profile + square_profile + triangle_profile
visualize_tensor(combined_profile)

# %%
# Single Lens Profile
# -------------------
# We can use a similar approach to create a profile with a lensarray, consisting of multiple lenses.
# Each lens will have a circular aperture and a specified focal length. Let's creat a single lens:

focal_length = 0.2
wavelength = 800e-9
spacing = 10e-6
radius = 1e-3

phase = lens_phase(shape, focal_length, wavelength, spacing)
visualize_tensor(torch.exp(1j * phase), title="Lens Phase")
# %%
circular_aperture = circle(shape, radius, spacing)
visualize_tensor(circular_aperture, title="Circular Aperture")

# %%
# Together:
visualize_tensor(circular_aperture * torch.exp(1j * phase))

# %%
# Lens Array
# -----------
# Now, let's create a lens array with in a 3x3 grid.
lens_array_profile = torch.zeros((shape, shape), dtype=torch.complex64)
for offset_i in [-2e-3, 0, 2e-3]:
    for offset_j in [-2e-3, 0, 2e-3]:
        offset = (offset_i, offset_j)
        phase = lens_phase(shape, focal_length, wavelength, spacing, offset)
        circular_aperture = circle(shape, radius, spacing, offset)

        lens_array_profile += circular_aperture * torch.exp(1j * phase)

visualize_tensor(lens_array_profile, title="Lens Array Profile")
