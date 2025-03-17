"""
Spatial Coherence
===================

Simulates Gaussian beams with low and high spatial coherence.
"""

# %%

import torch

import torchoptics
from torchoptics import CoherenceField
from torchoptics.profiles import gaussian_schell_model as gsm

# Set simulation properties
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

# Initialize coherence fields
low_coherence_field = CoherenceField(gsm(shape, waist_radius, low_coherence_width)).to(device)
high_coherence_field = CoherenceField(gsm(shape, waist_radius, high_coherence_width)).to(device)

# %%
# Propagation of Low and High Coherence Fields
# --------------------------------------------
# We propagate Gaussian-Schell model beams with low and high coherence over different distances.
# Low-coherence fields exhibit rapid changes in their spatial distribution, whereas high-coherence
# fields maintain their structure.

propagation_distances = [0, 0.01, 0.02]

for z in propagation_distances:
    low_coherence_field.propagate_to_z(z).visualize(title=f"Low Coherence Field at z = {z} m", vmin=0)
    high_coherence_field.propagate_to_z(z).visualize(title=f"High Coherence Field at z = {z} m", vmin=0)
