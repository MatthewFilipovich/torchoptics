"""
Grating Diffraction Orders
============================

Compares diffraction patterns from binary, blazed, and sinusoidal gratings.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens, PhaseModulator
from torchoptics.profiles import binary_grating, blazed_grating, gaussian, sinusoidal_grating

# %%
# Simulation Parameters
# ---------------------
# We define a focused optical system: a grating illuminated by a Gaussian beam, followed by
# a lens that maps the far-field diffraction pattern to its focal plane.

shape = 800  # Grid size
grating_period = 200e-6  # Grating period (m)
grating_height = torch.pi  # Phase modulation depth (radians)
focal_length = 0.2  # Lens focal length (m)
waist_radius = 1.5e-3  # Gaussian beam waist radius (m)

torchoptics.set_default_spacing(5e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Input Gaussian Beam
# --------------------
# A collimated Gaussian beam illuminates the grating.

input_field = Field(gaussian(shape, waist_radius))
input_field.visualize(title="Input Gaussian Beam")

# %%
# Binary Grating
# ---------------
# A binary (square-wave) phase grating alternates between 0 and a fixed phase depth.
# It produces symmetric diffraction orders: :math:`0, \pm 1, \pm 2, \ldots`

binary = binary_grating(shape, period=grating_period, height=grating_height)
binary_mod = PhaseModulator(binary, z=0)
binary_mod.visualize(title="Binary Phase Grating")

system = System(binary_mod, Lens(shape, focal_length, z=focal_length))
far_field = system.measure_at_z(input_field, z=2 * focal_length)
far_field.visualize(title="Binary Grating — Diffraction Pattern")

# %%
# Blazed Grating
# ---------------
# A blazed (sawtooth) phase grating concentrates most of the diffracted energy into
# a single order, making it much more efficient than a binary grating.

blazed = blazed_grating(shape, period=grating_period, height=grating_height)
blazed_mod = PhaseModulator(blazed, z=0)
blazed_mod.visualize(title="Blazed Phase Grating")

system = System(blazed_mod, Lens(shape, focal_length, z=focal_length))
far_field = system.measure_at_z(input_field, z=2 * focal_length)
far_field.visualize(title="Blazed Grating — Diffraction Pattern")

# %%
# Sinusoidal Grating
# -------------------
# A sinusoidal phase grating produces only the :math:`0` and :math:`\pm 1` orders
# (for small modulation depths), with intensity distributed according to Bessel functions.

sinusoidal = sinusoidal_grating(shape, period=grating_period, height=grating_height)
sinusoidal_mod = PhaseModulator(sinusoidal, z=0)
sinusoidal_mod.visualize(title="Sinusoidal Phase Grating")

system = System(sinusoidal_mod, Lens(shape, focal_length, z=focal_length))
far_field = system.measure_at_z(input_field, z=2 * focal_length)
far_field.visualize(title="Sinusoidal Grating — Diffraction Pattern")

# %%
# Rotated Grating
# ----------------
# Gratings can be rotated using the ``theta`` parameter. This tilts the diffraction
# orders in the Fourier plane.

angles = [0, torch.pi / 6, torch.pi / 4, torch.pi / 3]
for theta in angles:
    rotated = blazed_grating(shape, period=grating_period, height=grating_height, theta=theta)
    rotated_mod = PhaseModulator(rotated, z=0)

    system = System(rotated_mod, Lens(shape, focal_length, z=focal_length))
    far_field = system.measure_at_z(input_field, z=2 * focal_length)
    far_field.visualize(title=f"Blazed Grating at θ = {theta * 180 / torch.pi:.0f}°")
