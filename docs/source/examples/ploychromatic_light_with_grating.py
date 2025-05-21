"""Polychromatic Light with Grating.
====================================

Simulates the propagation of red, green, and blue light through a blazed grating.
"""

# %%

import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import PolychromaticPhaseModulator
from torchoptics.profiles import blazed_grating, gaussian

# %%
# Simulation Parameters
# ---------------------
# Define the grid size, wavelengths, grating properties, and propagation settings.

shape = 500  # Grid size (number of points per dimension)
waist_radius = 300e-6  # Waist radius of the Gaussian beam (m)
wavelengths = [450e-9, 550e-9, 700e-9]  # Blue, green, red wavelengths (m)
grating_period = 100e-6  # Period of the blazed grating (m)

# Select computation device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure torchoptics defaults
torchoptics.set_default_spacing(10e-6)

# %%
# Function to Plot RGB Intensities
# --------------------------------
# This function overlays the intensity distributions of the three fields
# as red, green, and blue channels.


def plot_rgb_intensities(fields, title) -> None:
    """Plots the RGB intensities of three fields as an RGB image."""
    if not isinstance(fields, list) or len(fields) != 3:
        msg = "`fields` must be a list of 3 Fields (R, G, B)."
        raise ValueError(msg)

    rgb_intensities = torch.stack([field.intensity() for field in fields], dim=-1).cpu().numpy()

    plt.figure()
    plt.imshow(rgb_intensities.clip(0, 1))
    plt.axis("off")
    plt.title(title)
    plt.show()


# %%
# Generating Polychromatic Gaussian Fields
# ----------------------------------------
# We create three Gaussian fields, each with a different wavelength (R, G, B).

gaussian_data = gaussian(shape, waist_radius).real
gaussian_data /= gaussian_data.max()  # Normalize to max intensity of 1

fields = [Field(gaussian_data, wavelength).to(device) for wavelength in wavelengths]

# Visualize the RGB intensity distribution of the input fields
plot_rgb_intensities(fields, title="Input RGB Fields")

# %%
# Blazed Grating Modulator
# ------------------------
# We apply a blazed grating phase modulation to the fields.

phase = blazed_grating(shape, grating_period, height=wavelengths[0])
system = System(PolychromaticPhaseModulator(phase)).to(device)

# Visualize the phase modulation at one of the wavelengths
system[0].visualize(wavelength=wavelengths[0], title="Blazed Grating Phase Modulation")

# %%
# Propagation Through the Grating System
# --------------------------------------
# We propagate the polychromatic fields through the system at different distances.

propagation_distances = torch.arange(0, 0.3, 0.05)

for z in propagation_distances:
    propagated_fields = [system.measure_at_z(field, z) for field in fields]
    plot_rgb_intensities(propagated_fields, title=f"z = {z:.2f} m")
