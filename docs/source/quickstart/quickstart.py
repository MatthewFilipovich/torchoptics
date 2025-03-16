"""
Quickstart
==========

Welcome to the TorchOptics quickstart! This guide walks you through the main concepts of optical
simulations, including:

- Implementing optical fields using the :class:`~torchoptics.Field` class
- Propagating fields through free space
- Simulating lenses for focusing and imaging
- Simplifying simulations with the :class:`~torchoptics.System` class

Before starting, make sure TorchOptics is installed (:ref:`installation`).
"""

# %%
# Import TorchOptics
# ------------------
#
# We first import the necessary TorchOptics components:

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens
from torchoptics.profiles import triangle

# %%
# Define Simulation Defaults
# --------------------------
#
# We set the default simulation parameters for the simulation:
#
# - ``spacing``: Physical separation between grid points for optical fields and elements, determining
#   simulation resolution.
# - ``wavelength``: Wavelength of the simulated monochromatic optical field.

torchoptics.set_default_spacing(10e-6)  # 10 um
torchoptics.set_default_wavelength(700e-9)  # 700 nm

# %%
# Initialize Optical Field
# ------------------------
#
# Monochromatic optical fields are represented in TorchOptics using the :class:`~torchoptics.Field` class,
# which encapsulates the complex-valued wavefronts sampled along the :math:`xy`-plane.
#
# Let's create an optical field with a triangular amplitude profile:

shape = 500  # Grid shape (500x500 points)
base = 2e-3  # Triangle base width (2 mm)
height = 1e-3  # Triangle height (1 mm)

# Generate triangular profile and initialize the field
triangle_profile = triangle(shape, base, height)
field = Field(triangle_profile)

# Visualize the field distribution
field.visualize(title="Initial Field at z=0 m")
print(field)

# %%
# Free-Space Propagation
# ----------------------
#
# Next, we'll simulate the propagation of the optical field in free space to :math:`z=0.1` m and visualize the
# diffracted field:

propagated_field = field.propagate_to_z(0.1)
propagated_field.visualize(title="Propagated Field at z=0.1 m")

# %%
# Image Formation with a Lens
# -----------------------------
#
# We'll now focus the optical field using a lens to form an image. The lens is modeled as a thin lens
# with a specified focal length :math:`f`, which relates the object distance :math:`d_o` and image distance
# :math:`d_i` through the thin lens equation:
#
# .. math::
#     \frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}
#
# We'll set the following parameters:

focal_length = 0.2  # Lens focal length (20 cm)
d_o = 0.4  # Object-to-lens distance (40 cm)
d_i = 0.4  # Lens-to-image distance (40 cm)

lens_z = d_o  # Position of the lens along the z-axis
image_z = lens_z + d_i  # Position of the image plane along the z-axis

print(f"Lens Position: {lens_z} m")
print(f"Image Plane Position: {image_z} m")

# %%
# Initialize the Lens
# ^^^^^^^^^^^^^^^^^^^
#
# We create a lens using the :class:`~torchoptics.elements.Lens` class, which modulates the field with a
# quadratic phase factor and applies a circular aperture:

lens = Lens(shape, focal_length, lens_z)
lens.visualize(title="Lens Profile")
print(lens)


# %%
# Field at Lens Plane
# ^^^^^^^^^^^^^^^^^^^^
#
# Let's propagate the field to the lens :math:`z`-position:

field_before_lens = field.propagate_to_z(lens_z)
field_before_lens.visualize(title="Field Before Lens")

# %%
#
# Next, we apply the lens transformation to the field:
field_after_lens = lens(field_before_lens)
field_after_lens.visualize(title="Field After Lens")

# %%
# Field at the Image Plane
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Finally, we'll propagate the field from the lens position to the image plane:

field_image_plane = field_after_lens.propagate_to_z(image_z)
field_image_plane.visualize(title="Field at Image Plane")

# %%
# System Class
# -------------
#
# The :class:`~torchoptics.System` class simplifies optical simulations by automatically handling field
# propagation and element interactions, removing the need for manual updates at each step:

system = System(lens)
print(system)

# %%
# Measure at Image Plane
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We can measure the field at a specific z-plane using the :meth:`~torchoptics.System.measure_at_z` method:
field_image_plane = system.measure_at_z(field, z=image_z)
field_image_plane.visualize(title="Field at Image Plane (System)")
