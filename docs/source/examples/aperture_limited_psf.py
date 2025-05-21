"""Aperture-Limited PSF.
=====================

Calculates the point spread function (PSF) of a point source imaged by a lens with various aperture diameters.
"""

# %%
import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Modulator
from torchoptics.profiles import circle, lens_phase

# %%
# Simulation Setup
# ----------------
# Define grid spacing and wavelength used in the simulation.

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Generate Input Point Source
# ---------------------------
# A delta function simulates an ideal point source at the center of the grid.

shape = 500
input_profile = torch.zeros(shape, shape)
input_profile[shape // 2, shape // 2] = 1.0  # Delta function at the center
input_field = Field(input_profile)

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

focal_length = 0.5  # 50 cm
d_o = 1  # 1 m
d_i = 1  # 1 m

lens_z = d_o
image_z = d_o + d_i


# %%
# Resolution vs. Aperture Diameter
# --------------------------------
# We simulate imaging a point source with several lens aperture diameters.
# The resulting point spread function (PSF) shows how diffraction limits resolution.

aperture_diameters = [5e-3, 4e-3, 3e-3, 2e-3, 1e-3]  # Diameters in meters

for diameter in aperture_diameters:
    label = f"{diameter * 1e3:.0f} μm"

    # Circular aperture at lens plane
    amplitude = circle(shape, diameter / 2)
    phase = lens_phase(shape, focal_length)
    lens = Modulator(amplitude * torch.exp(1j * phase), z=lens_z)
    lens.visualize(title=f"Aperture Mask (Diameter = {label})")

    # Build optical system
    system = System(lens)

    # Propagate field to image plane
    field_image = system.measure_at_z(input_field, z=image_z)

    # Visualize output (PSF)
    field_image.visualize(title=f"Point Spread Function (Aperture Diameter = {label})")
