"""
Aperture-Limited Resolution
============================

Simulates how the aperture size of a thin lens affects spatial resolution using a Siemens star test pattern.
"""

# %%
import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens, Modulator
from torchoptics.profiles import circle, siemens_star

# %%
# Simulation Setup
# ----------------
# Define grid spacing and wavelength used in the simulation.

torchoptics.set_default_spacing(10e-6)  # 10 μm grid spacing
torchoptics.set_default_wavelength(700e-9)  # 700 nm wavelength

# %%
# Generate Siemens Star Test Pattern
# ----------------------------------

shape = 500
star_radius = 1e-3
num_spokes = 30

input_profile = siemens_star(shape, num_spokes, star_radius)
input_field = Field(input_profile)
input_field.visualize(title="Input Field: Siemens Star", interpolation="none")

# %%
# Define Imaging Geometry
# -----------------------
# We simulate image formation using a thin lens. The imaging condition is given by:
#
# .. math::
#     \frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}.
#
# where:
#
# - :math:`f` is the focal length,
# - :math:`d_o` is the object distance (from source to lens),
# - :math:`d_i` is the image distance (from lens to detector).

focal_length = 0.1  # 10 cm
d_o = 0.2  # 20 cm
d_i = 0.2  # 20 cm

lens_z = d_o
image_z = d_o + d_i

print(f"Lens position: z = {lens_z} m")
print(f"Image plane: z = {image_z} m")

# %%
# Resolution vs. Aperture Diameter
# --------------------------------
# We simulate imaging with several lens aperture diameters and observe their impact on resolution.
# Smaller apertures introduce diffraction blur, which reduces the ability to resolve fine details.

aperture_diameters = [4e-3, 2e-3, 1e-3]  # Diameters in meters

for diameter in aperture_diameters:
    label = f"{diameter * 1e3:.0f} μm"

    # Circular aperture at lens plane
    aperture = Modulator(circle(shape, radius=diameter / 2), z=lens_z)
    aperture.visualize(title=f"Aperture Mask (Diameter = {label})")

    # Lens at same plane
    lens = Lens(shape, focal_length, z=lens_z)

    # Build optical system
    system = System(lens, aperture)

    # Propagate field to image plane
    field_image = system.measure_at_z(input_field, z=image_z)

    # Visualize output
    field_image.visualize(title=f"Image Formed (Aperture Diameter = {label})", interpolation="none", vmax=1)
