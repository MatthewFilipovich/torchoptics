"""
Quickstart
==========

Welcome to the TorchOptics quickstart! This guide walks you through the fundamental concepts of optical
simulations, including:

- Creating an optical field
- Propagating it through free space
- Simulating an imaging system with a single lens

Before starting, make sure TorchOptics is installed (:ref:`installation`).
"""

# %%
# Import Modules
# ---------------
#
# We first import the necessary TorchOptics components:

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens
from torchoptics.profiles import triangle

# %%
# Set Defaults
# -------------
#
# We specify the simulation default parameters, which include:
#
# - ``spacing``: Specifies the physical distance between adjacent points in the xy-plane of the simulation
#   grid. This controls the spatial resolution of optical fields and elements, affecting
#   numerical accuracy and computational cost.
#
# - ``wavelength``: Defines the wavelength of the monochromatic light used in the simulation.

torchoptics.set_default_spacing(10e-6)  # Spatial grid spacing (m)
torchoptics.set_default_wavelength(700e-9)  # Default wavelength of the field (m)

# %%
# Initialize Optical Field
# ------------------------
#
# Monochromatic optical fields are represented in TorchOptics using the :class:`~torchoptics.Field` class,
# which encapsulates the complex-valued wavefronts sampled along the xy-plane.
#
# Let's create an optical field with a triangular amplitude profile.

shape = 500  # Define the shape of the field along the x and y axes (number of points)
base = 2e-3  # Base length of the triangle profile (m)
height = 1e-3  # Height of the triangle profile (m)

# Generate a triangle-shaped optical profile as a 2D PyTorch tensor.
triangle_profile = triangle(shape, base, height)

# Create the optical field using the triangle profile.
field = Field(triangle_profile)

# Visualize the field and print its properties.
field.visualize(title="Initial Triangle Field")
print(field)

# %%
# Free-Space Propagation
# ----------------------
#
# Next, we'll simulate the propagation of the optical field in free space. This is performed using the
# :meth:`~torchoptics.Field.propagate_to_z` method, which returns the field at aspecified distance along the
# z-axis.

propagated_field = field.propagate_to_z(0.1)
propagated_field.visualize(title="Propagated Field at z=0.1 m")

# %%
# Define Lens Parameters
# ----------------------
#
# A thin lens introduces a quadratic phase factor, effectively modifying the curvature of the optical field.
# Its focusing behavior is described by the lensmaker’s formula (also called the thin-lens equation):
#
# .. math::
#
#     \frac{1}{f} = \frac{1}{z_o} + \frac{1}{z_i}
#
# where:
#
# - :math:`f` is the focal length of the lens.
# - :math:`z_o` is the distance from the lens to the object plane (input field).
# - :math:`z_i` is the distance from the lens to the image plane, at which the field is ideally refocused.
#
# In this example, we place the lens and define object/image planes based on these distances.

focal_length = 0.2  # Lens focal length (m)
d_o = 0.4  # Object-to-lens distance (m)
d_i = 0.4  # Lens-to-image distance (m)

lens_z = d_o  # Absolute lens position along the z-axis (m)
image_z = lens_z + d_i  # Absolute image-plane position along the z-axis (m)

print(f"Lens Position: {lens_z} m")
print(f"Image Plane Position: {image_z} m")

# %%
# Initialize Lens
# ---------------

lens = Lens(shape, focal_length, lens_z)
print(lens)  # Display the lens properties


# %%
# Field Before the Lens
# ---------------------
#
# Before applying the lens, let's propagate the field to the lens position and visualize its distribution.

field_before_lens = field.propagate_to_z(lens_z)
field_before_lens.visualize(title="Field Before Lens")

# %%
# Field After the Lens
# ---------------------
#
# Next, apply the lens transformation. This step lets us observe how the lens immediately reshapes the optical field.
field_after_lens = lens(field_before_lens)
field_after_lens.visualize(title="Field After Lens")

# %%
# Field at the Image Plane
# ------------------------
#
# Finally, we'll propagate the field from the lens position to the image plane. This step shows how well the lens focuses the field.

field_image_plane = field_after_lens.propagate_to_z(image_z)
field_image_plane.visualize(title="Field at Image Plane")

# %%
# System Class
# -------------
#
# Instead of manually applying each optical element, the :class:`~torchoptics.System` class lets you define
# a full optical setup in a structured way. This simplifies complex simulations and makes it easy to add or remove elements.

system = System(lens)
print(system)  # Display the system's properties

# %%
# Measure at Image Plane
# -----------------------
field_image_plane = system.measure_at_z(field, z=image_z)
field_image_plane.visualize(title="Image Plane Using System Class")
