"""
Zernike Aberrations
====================

Simulates the effect of Zernike wavefront aberrations on a focused beam.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import Lens, PhaseModulator
from torchoptics.profiles import gaussian, zernike

# %%
# Simulation Parameters
# ---------------------
# We set up a focused Gaussian beam and apply different Zernike aberrations at the lens.

shape = 400  # Grid size
focal_length = 0.2  # Focal length (m)
waist_radius = 800e-6  # Beam waist radius (m)
aberration_radius = 2e-3  # Radius of the Zernike aperture (m)
aberration_strength = 3.0  # Peak phase of aberration (radians)

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Unaberrated Focus
# -----------------
# First, we focus a Gaussian beam with a perfect lens to establish the ideal point
# spread function (PSF).

input_field = Field(gaussian(shape, waist_radius))

system = System(Lens(shape, focal_length, z=focal_length))
ideal_focus = system.measure_at_z(input_field, z=2 * focal_length)
ideal_focus.visualize(title="Ideal Focus (No Aberration)")

# %%
# Zernike Aberration Gallery
# --------------------------
# Zernike polynomials are the standard basis for describing wavefront aberrations.
# Each polynomial :math:`Z_n^m` is characterized by radial order :math:`n` and
# azimuthal order :math:`m`.
#
# We apply different Zernike aberrations and observe their effect on the focal spot.

aberrations = [
    (2, 0, "Defocus"),
    (2, 2, "Astigmatism (Vertical)"),
    (2, -2, "Astigmatism (Oblique)"),
    (3, 1, "Coma (Vertical)"),
    (3, -1, "Coma (Horizontal)"),
    (4, 0, "Spherical Aberration"),
]

for n, m, name in aberrations:
    # Generate the Zernike polynomial as a phase aberration
    z_phase = aberration_strength * zernike(shape, n, m, aberration_radius)

    # Create a phase modulator with the aberration at the lens plane
    aberration = PhaseModulator(z_phase, z=focal_length)

    # Build the system: lens + aberration at the same plane
    system = System(
        Lens(shape, focal_length, z=focal_length),
        aberration,
    )

    # Visualize the aberration phase
    aberration.visualize(title=f"Aberration: {name} ($Z_{{{n}}}^{{{m}}}$)")

    # Measure the aberrated PSF
    aberrated_focus = system.measure_at_z(input_field, z=2 * focal_length)
    aberrated_focus.visualize(title=f"PSF with {name}")
