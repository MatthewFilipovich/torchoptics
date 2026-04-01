"""
Animate Propagation
===================

Creates an animation of a beam diffracting as it propagates through free space.
An octagonal aperture illuminated by a plane wave develops increasingly complex
diffraction patterns at larger distances, illustrating the transition from the
Fresnel (near-field) to Fraunhofer (far-field) regime.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_end_ignore

import torch

import torchoptics
from torchoptics import Field, animate_tensor
from torchoptics.profiles import octagon

# %%
# Simulation Parameters
# ---------------------

shape = 500  # Grid size
spacing = 10e-6  # Grid spacing (m)
wavelength = 700e-9  # Red light (m)
aperture_radius = 1.5e-3  # Octagon circumradius (m)

# Configure torchoptics defaults
torchoptics.set_default_spacing(spacing)
torchoptics.set_default_wavelength(wavelength)

# %%
# Input Field: Octagonal Aperture
# --------------------------------
# A uniform plane wave transmitted through an octagonal aperture serves as the
# input. The sharp edges produce rich diffraction features during propagation.

field = Field(octagon(shape, radius=aperture_radius))
field.visualize(title="Octagonal Aperture (z = 0)")

# %%
# Propagation Animation
# ---------------------
# We propagate the field from :math:`z = 0` to :math:`z = 2` m and collect the
# intensity at each plane. The animation reveals how the sharp aperture edges
# produce Fresnel fringes that gradually evolve into the far-field pattern.

z_values = torch.linspace(0, 2, 101)
intensities = torch.stack([field.propagate_to_z(z).intensity() for z in z_values])

titles = [f"z = {z:.2f} m" for z in z_values]
animate_tensor(intensities, vmax=2, title=titles, func_anim_kwargs={"interval": 100})
