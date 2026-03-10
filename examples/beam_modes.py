"""
Beam Modes
==========

Visualizes Hermite-Gaussian, Laguerre-Gaussian, and Bessel beam modes and their propagation.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import torchoptics
from torchoptics import Field
from torchoptics.profiles import bessel, gaussian, hermite_gaussian, laguerre_gaussian

# %%
# Simulation Parameters
# ---------------------
# We set the grid size, spacing, and wavelength for the simulation.

shape = 300  # Grid size (number of points per dimension)
waist_radius = 500e-6  # Beam waist radius (m)

torchoptics.set_default_spacing(5e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Fundamental Gaussian Beam
# -------------------------
# The fundamental Gaussian beam (TEM₀₀) has a simple bell-shaped intensity profile.
# It is the lowest-order solution of the paraxial wave equation.

gauss = Field(gaussian(shape, waist_radius))
gauss.visualize(title="Gaussian Beam (TEM₀₀)")

# %%
# Hermite-Gaussian Modes
# ----------------------
# Hermite-Gaussian (HG) modes are characterized by two mode indices :math:`(m, n)`.
# The fundamental Gaussian is HG₀₀. Higher-order modes have nodal lines along
# the :math:`x` and :math:`y` axes.

for m, n in [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 2)]:
    hg = Field(hermite_gaussian(shape, m, n, waist_radius))
    hg.visualize(title=f"HG$_{{{m}{n}}}$")

# %%
# Laguerre-Gaussian Modes
# -----------------------
# Laguerre-Gaussian (LG) modes are characterized by radial index :math:`p` and azimuthal
# index :math:`l`. Modes with :math:`l \neq 0` carry orbital angular momentum and have a
# helical phase front.

for p, l in [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (2, 0)]:
    lg = Field(laguerre_gaussian(shape, p, l, waist_radius))
    lg.visualize(title=f"LG$_{{{p}{l}}}$")

# %%
# Bessel Beam
# -----------
# Bessel beams are non-diffracting solutions characterized by a cone angle. They feature a
# central bright spot surrounded by concentric rings of equal peak intensity.

cone_angle = 0.005  # Cone angle (radians)
b = Field(bessel(shape, cone_angle))
b.visualize(title=f"Bessel Beam (cone angle = {cone_angle} rad)")

# %%
# Propagation of Beam Modes
# -------------------------
# We compare the propagation of a Gaussian beam and a Bessel beam. The Gaussian beam
# diverges, while the Bessel beam maintains its transverse structure.

gauss_field = Field(gaussian(shape, waist_radius))
bessel_field = Field(bessel(shape, cone_angle))

propagation_distances = [0, 0.05, 0.1, 0.15]
for z in propagation_distances:
    gauss_field.propagate_to_z(z).visualize(title=f"Gaussian at z = {z} m")
    bessel_field.propagate_to_z(z).visualize(title=f"Bessel at z = {z} m")
