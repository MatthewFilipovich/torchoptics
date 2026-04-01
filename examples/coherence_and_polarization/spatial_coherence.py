"""
Spatial Coherence
=================

Simulates partially coherent Gaussian beams using the Gaussian-Schell model (GSM).
Spatial coherence describes the correlation of a field at two different spatial
points. In the GSM, the coherence width :math:`\\sigma_c` controls how correlated
the field is across the beam profile:

.. math::
    \\Gamma(\\mathbf{r}_1, \\mathbf{r}_2) = \\sqrt{I(\\mathbf{r}_1)\\, I(\\mathbf{r}_2)}
    \\; \\exp\\!\\left(-\\frac{|\\mathbf{r}_1 - \\mathbf{r}_2|^2}{2\\sigma_c^2}\\right)

When :math:`\\sigma_c \\gg w_0` the field is nearly fully coherent; when
:math:`\\sigma_c \\ll w_0` it is highly incoherent and diverges rapidly
upon propagation.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_end_ignore

import torch

import torchoptics
from torchoptics import SpatialCoherence
from torchoptics.profiles import gaussian_schell_model as gsm

# %%
# Simulation Parameters
# ---------------------
# We compare two beams with identical intensity profiles but different
# coherence widths. The coherence width relative to the beam waist determines
# the divergence behavior during propagation.

shape = 30  # Grid size (number of points per dimension)
spacing = 10e-6  # Grid spacing (m)
wavelength = 700e-9  # Wavelength (m)
waist_radius = 40e-6  # Waist radius of the Gaussian beam (m)

# Define coherence widths
low_coherence_width = 10e-6  # Low spatial coherence (m)
high_coherence_width = 1e-3  # High spatial coherence (m)

# Determine computation device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure torchoptics defaults
torchoptics.set_default_spacing(spacing)
torchoptics.set_default_wavelength(wavelength)

print(f"Beam waist: {waist_radius * 1e6:.0f} µm")
low_ratio = low_coherence_width / waist_radius
print(f"Low coherence width: {low_coherence_width * 1e6:.0f} µm (σ_c/w₀ = {low_ratio:.2f})")
high_ratio = high_coherence_width / waist_radius
print(f"High coherence width: {high_coherence_width * 1e3:.1f} mm (σ_c/w₀ = {high_ratio:.0f})")

# %%
# Coherence Matrices
# ------------------
# The Gaussian-Schell model generates a mutual coherence matrix that encodes
# both the intensity profile and the spatial correlations. We initialize two
# :class:`~torchoptics.SpatialCoherence` objects representing low- and
# high-coherence beams.

low_spatial_coherence = SpatialCoherence(gsm(shape, waist_radius, low_coherence_width)).to(device)
high_spatial_coherence = SpatialCoherence(gsm(shape, waist_radius, high_coherence_width)).to(device)

# %%
# Propagation Comparison
# ----------------------
# Low-coherence fields spread rapidly and lose structure, while high-coherence
# fields maintain their spatial distribution over longer distances, analogous
# to how a laser beam (high coherence) stays collimated while a thermal source
# (low coherence) diverges quickly.

propagation_distances = [0, 0.01, 0.02]

for z in propagation_distances:
    low_spatial_coherence.propagate_to_z(z).visualize(title=f"Low Coherence at z = {z} m", vmin=0)
    high_spatial_coherence.propagate_to_z(z).visualize(title=f"High Coherence at z = {z} m", vmin=0)
