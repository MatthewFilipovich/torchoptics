"""
Quickstart
===========

This tutorial demonstrates the basic functionalities of the TorchOptics library,
including setting up simulations, propagating optical fields, and using lenses.
"""

# %%
import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens
from torchoptics.profiles import triangle

# %%
# Setting Up TorchOptics
# ----------------------
#
# Before defining optical fields, we must specify default grid spacing and wavelength.
# These values affect all subsequent operations and should be chosen based on the
# physical scale of your simulation.

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

shape = 500
base_length = 2e-3  # Base length of the triangle profile (m)
triangle_height = 1e-3  # Height of the triangle profile (m)

triangle_profile = triangle(shape, base_length, triangle_height)
field = Field(triangle_profile)

field.visualize(title="Initial Triangle Field")

# %%
# Free-Space Propagation
# ----------------------
#
# Light propagates freely in space according to the Huygens-Fresnel principle.
# Here, we compute the field at a distance of 0.1 meters from its initial position.

propagated_field = field.propagate_to_z(0.1)
propagated_field.visualize(title="Propagated Field at z=0.1 m")

# %%
# Introducing a Lens
# ------------------
#
# A lens focuses or diverges light according to its focal length. Using the thin-lens equation:
#
# .. math::
#
#    \frac{1}{f} = \frac{1}{z} + \frac{1}{z'}
#
# where :math:`f` is the focal length, :math:`z` is the distance from the object to the lens,
# and :math:`z'` is the image distance.

focal_length = 0.2  # Focal length of the lens (m)
lens_z = 0.4  # Position of the lens along the z-axis (m)
image_z = 0.8  # Position of the image plane (m)

# %%
# Field Before Lens
# -----------------
#
# Propagate the field to the lens position to observe its state just before encountering the lens.

field_before_lens = field.propagate_to_z(lens_z)
field_before_lens.visualize(title="Field Before Lens")

# %%
# Field After Lens
# ----------------
#
# Apply the lens transformation to see how the field is modified immediately after passing through the lens.

lens = Lens(shape, focal_length, lens_z)
field_after_lens = lens(field_before_lens)
field_after_lens.visualize(title="Field After Lens")

# %%
# Field at Image Plane
# --------------------
#
# Finally, propagate the field from the lens to the image plane to observe the resulting field distribution.

field_image_plane = field_after_lens.propagate_to_z(image_z)
field_image_plane.visualize(title="Field at Image Plane")

# %%
# Using the System Class
# ----------------------
#
# The `System` class allows us to build a sequence of optical elements and measure the field
# at different points without manually computing each step. This simplifies complex simulations.

system = System(lens)
field_image_plane = system.measure_at_z(field, z=image_z)
field_image_plane.visualize(title="Image Plane via System Class")
