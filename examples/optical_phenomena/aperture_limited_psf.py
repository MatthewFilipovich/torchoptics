"""
Aperture-Limited PSF
=====================

Calculates the point spread function (PSF) of a lens imaging system as a function
of aperture diameter. The PSF defines the fundamental resolution limit: larger
apertures produce sharper Airy-disk PSFs, while smaller apertures broaden the PSF
due to increased diffraction. This tradeoff is described by the Rayleigh criterion:

.. math::
    \\theta_{\\text{min}} = 1.22 \\frac{\\lambda}{D}

where :math:`D` is the aperture diameter. In terms of physical size at the image
plane, the Rayleigh resolution radius is:

.. math::
    r_{\\text{Rayleigh}} = 1.22 \\frac{\\lambda f}{D}

where :math:`f` is the focal length.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Modulator
from torchoptics.profiles import circle, lens_phase

# %%
# Simulation Setup
# ----------------
# We use a point source at z = 0, a lens at z = 1 m, and the image plane at
# z = 2 m. The focal length is 0.5 m, so the object and image distances satisfy
# the thin-lens imaging condition.

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

shape = 500
wavelength = 700e-9  # m
spacing = 10e-6  # m

focal_length = 0.5  # m
d_o = 1.0  # Object distance (m)
d_i = 1.0  # Image distance (m), satisfies 1/f = 1/d_o + 1/d_i
lens_z = d_o
image_z = d_o + d_i

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Focal length: {focal_length} m")
print(f"Object distance: {d_o} m, Image distance: {d_i} m")
print(f"Wavelength: {wavelength * 1e9:.0f} nm")

# %%
# Point Source Input
# ------------------
# A delta function at the grid center simulates an ideal on-axis point source.

point_source_data = torch.zeros(shape, shape)
point_source_data[shape // 2, shape // 2] = 1.0
input_field = Field(point_source_data).to(device)

# %%
# PSF Gallery: Effect of Aperture Diameter
# -----------------------------------------
# We compute the PSF for five aperture diameters and display them side-by-side.
# The theoretical Rayleigh radius (first dark ring of the Airy pattern) scales
# inversely with aperture diameter.

aperture_diameters = [5e-3, 4e-3, 3e-3, 2e-3, 1e-3]  # m
phase = lens_phase(shape, focal_length)

fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), constrained_layout=True)

for ax, diameter in zip(axes, aperture_diameters):
    # Build aperture + lens element
    amplitude = circle(shape, diameter / 2)
    lens = Modulator(amplitude * torch.exp(1j * phase), z=lens_z)
    system = System(lens).to(device)

    # Propagate to image plane
    psf_field = system.measure_at_z(input_field, z=image_z)
    psf = psf_field.intensity().cpu()
    ax.imshow(psf, cmap="inferno")
    ax.set_title(f"Aperture: {diameter * 1e3:.1f} mm")
    ax.axis("off")

plt.suptitle("Aperture-Limited PSF", fontsize=16)
plt.show()
