"""Training Diffractive Splitter.
==============================

Trains a diffractive optical system to function as a diffractive splitter.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import torch
from torch.nn import Parameter

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import PhaseModulator
from torchoptics.profiles import gaussian

# %%
# Simulation Parameters
# ---------------------
# Define the grid size and beam properties.

shape = 250  # Grid size (number of points per dimension)
waist_radius = 150e-6  # Waist radius of the Gaussian beam (m)

# Select computation device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure torchoptics defaults
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Target Field: Diffractive Splitting
# -----------------------------------
# We define the target field consisting of four Gaussian spots arranged in a square pattern.

offset = 3.8 * waist_radius  # Offset for Gaussian spots
target_field_data = (
    gaussian(shape, waist_radius, offset=(offset, offset))
    + gaussian(shape, waist_radius, offset=(offset, -offset))
    + gaussian(shape, waist_radius, offset=(-offset, offset))
    + gaussian(shape, waist_radius, offset=(-offset, -offset))
) / 2  # Normalize the intensity

target_field = Field(target_field_data, z=0.6).to(device)
target_field.visualize(title="Target Field")

# %%
# Input Field: Single Gaussian Beam
# ---------------------------------
# The input field is a single Gaussian beam at :math:`z=0` m.

input_field = Field(gaussian(shape, waist_radius), z=0).to(device)
input_field.visualize(title="Input Field")

# %%
# Diffractive Optical System
# --------------------------
# The system consists of three trainable phase modulation layers.

system = System(
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.0),
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.2),
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.4),
).to(device)

# %%
# Training the System
# -------------------
# We optimize the phase modulators to transform the input field into the target field.

optimizer = torch.optim.Adam(system.parameters(), lr=0.1)
losses = []

for iteration in range(200):
    optimizer.zero_grad()
    output_field = system.measure_at_z(input_field, 0.6)
    loss = 1 - output_field.inner(target_field).abs().square()  # Match to target field
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if iteration % 10 == 0:
        pass

# %%
# Loss Curve
# ----------
# We plot the loss function to monitor training progress.

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss vs Iteration")
plt.yscale("log")
plt.xlim(0, len(losses))
plt.show()

# %%
# Visualizing the Trained Phase Modulators
# ----------------------------------------
# We inspect the phase modulation layers after training.

for i, element in enumerate(system):
    element.visualize(title=f"Phase Modulator {i + 1}")

# %%
# Output Field After Training
# ---------------------------
# Finally, we visualize the output field at the target plane (:math:`z = 0.6` m).

output_field = system.measure_at_z(input_field, 0.6)
output_field.visualize(title="Output Field")
