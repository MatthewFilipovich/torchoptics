"""
Telescope System
=================

Simulates a Keplerian telescope that magnifies an input field.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens
from torchoptics.profiles import siemens_star

# %%
# Simulation Parameters
# ---------------------
# A Keplerian telescope consists of two positive lenses separated by the sum of their
# focal lengths :math:`f_1 + f_2`. The magnification is :math:`M = -f_2 / f_1`.
#
# We simulate a 3× magnifying telescope using a short focal length objective and a
# longer focal length eyepiece.

shape = 600  # Grid size
f1 = 50e-3  # Objective focal length (50 mm)
f2 = 150e-3  # Eyepiece focal length (150 mm)
magnification = -f2 / f1  # Angular magnification

torchoptics.set_default_spacing(5e-6)
torchoptics.set_default_wavelength(550e-9)

print(f"Objective focal length: {f1 * 1e3:.0f} mm")
print(f"Eyepiece focal length: {f2 * 1e3:.0f} mm")
print(f"Magnification: {magnification:.1f}×")

# %%
# Input Field: Siemens Star Target
# ---------------------------------
# We use a Siemens star resolution target as the input field. This pattern has radial
# spokes that test resolution at all orientations.

target = siemens_star(shape, num_spokes=36, radius=600e-6)
input_field = Field(target)
input_field.visualize(title="Input: Siemens Star Target")

# %%
# Telescope Optical System
# ------------------------
# The objective lens is at :math:`z = f_1` from the object, and the eyepiece is
# at :math:`z = f_1 + f_1 + f_2 = 2f_1 + f_2` (separated from the objective by :math:`f_1 + f_2`).

objective_z = f1
eyepiece_z = f1 + f1 + f2
image_z = eyepiece_z + f2

system = System(
    Lens(shape, f1, z=objective_z),
    Lens(shape, f2, z=eyepiece_z),
)
print(system)

# %%
# Field at Key Planes
# --------------------
# We measure the field at key planes along the telescope:
#
# 1. **Input plane** (:math:`z = 0`): Original Siemens star
# 2. **Intermediate focus** (:math:`z = 2f_1`): Inverted real image
# 3. **Output plane** (:math:`z = 2f_1 + 2f_2`): Final magnified image

# Intermediate focal plane (between the two lenses)
intermediate_z = 2 * f1
intermediate = system.measure_at_z(input_field, z=intermediate_z)
intermediate.visualize(title="Intermediate Focus")

# Final image plane
output = system.measure_at_z(input_field, z=image_z)
output.visualize(title=f"Telescope Output ({magnification:.1f}× Magnification)")

# %%
# Resolution Comparison at Multiple Planes
# -----------------------------------------
# We measure the field at several planes along the optical axis to visualize
# how the image forms through the telescope.

import torch

z_values = torch.linspace(0, float(image_z), 7)
for z in z_values:
    field_at_z = system.measure_at_z(input_field, z=z.item())
    field_at_z.visualize(title=f"z = {z.item() * 1e3:.1f} mm")
