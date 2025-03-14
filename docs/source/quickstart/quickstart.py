"""
Quickstart
==========

In this quickstart tutorial, you'll learn the basics of working with the TorchOptics library.
We'll define optical fields, simulate their propagation through free space, and explore how lenses affect these fields.
"""

# %%
import torchoptics  # Core TorchOptics functionalities
from torchoptics import Field, System  # Fundamental classes for optical fields and optical systems
from torchoptics.elements import Lens  # Optical element: lens
from torchoptics.profiles import triangle  # Profile-generating function for triangle shapes

# %%
# Setting Up Your Simulation
# --------------------------
#
# Start by specifying some global settings. These will define the scale and resolution for your optical simulations.
# We use :func:`torchoptics.set_default_spacing` and :func:`torchoptics.set_default_wavelength` to set these parameters.

torchoptics.set_default_spacing(10e-6)  # Spatial grid spacing (m)
torchoptics.set_default_wavelength(700e-9)  # Default wavelength of the field (m)

# %%
# Let's create an optical field shaped like a triangle. We'll use the built-in profile function
# :func:`~torchoptics.profiles.triangle` to easily generate this shape.
shape = 500
base_length = 2e-3  # Base length of the triangle profile (m)
triangle_height = 1e-3  # Height of the triangle profile (m)

triangle_profile = triangle(shape, base_length, triangle_height)
field = Field(triangle_profile)

field.visualize(title="Initial Triangle Field")
print(field)  # Display the field's properties

# %%
# Propagating Through Free Space
# ------------------------------
#
# Next, we'll simulate how the optical field changes as it travels through free space.
# The method :meth:`~torchoptics.Field.propagate_to_z` handles this propagation based on physical principles.

propagated_field = field.propagate_to_z(0.1)
propagated_field.visualize(title="Propagated Field at z=0.1 m")

# %%
# Adding a Lens to the Simulation
# -------------------------------
#
# Now, let's see how adding a lens affects the field. A lens can focus or diverge light, and its behavior is described by the thin-lens equation:
#
# .. math::
#
#    \frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}
#
# Here, :math:`f` is the lens's focal length, :math:`d_o` is the distance from the object plane (which we've set at z=0),
# and :math:`d_i` is the distance from the lens to the resulting image plane.

focal_length = 0.2  # Lens focal length (m)
d_o = 0.4  # Object-to-lens distance (m)
d_i = 0.4  # Lens-to-image distance (m)

lens_z = d_o  # Absolute lens position along the z-axis (m)
image_z = lens_z + d_i  # Absolute image-plane position along the z-axis (m)

print(f"Lens Position: {lens_z} m")
print(f"Image Plane Position: {image_z} m")

# %%
# Checking the Field Before the Lens
# ----------------------------------
#
# Before applying the lens, let's propagate the field to the lens position and see its distribution at that point.

field_before_lens = field.propagate_to_z(lens_z)
field_before_lens.visualize(title="Field Before Lens")

# %%
# Observing the Field After the Lens
# ----------------------------------
#
# Next, apply the lens transformation. This step lets us observe how the lens immediately reshapes the optical field.

lens = Lens(shape, focal_length, lens_z)
print(lens)  # Display the lens properties

# %%
# Apply the lens to the field at the lens position
field_after_lens = lens(field_before_lens)
field_after_lens.visualize(title="Field After Lens")

# %%
# Examining the Field at the Image Plane
# --------------------------------------
#
# Finally, we'll propagate the field from the lens position to the image plane. This step shows how well the lens focuses the field.

field_image_plane = field_after_lens.propagate_to_z(image_z)
field_image_plane.visualize(title="Field at Image Plane")

# %%
# Simplifying with the System Class
# ---------------------------------
#
# TorchOptics provides the convenient :class:`~torchoptics.System` class, making it easy to chain multiple optical elements together.
# Let's quickly redo the previous steps using this simplified approach.

system = System(lens)
print(system)  # Display the system's properties

# %%
# Propagating the Field through the System
field_image_plane = system.measure_at_z(field, z=image_z)
field_image_plane.visualize(title="Image Plane Using System Class")
