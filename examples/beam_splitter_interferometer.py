"""
Beam Splitter Interferometer
=============================

Simulates a Mach-Zehnder interferometer using beam splitters and phase modulation.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field
from torchoptics.elements import BeamSplitter
from torchoptics.profiles import gaussian

# %%
# Simulation Parameters
# ---------------------
# We set up a Gaussian beam as the input to a Mach-Zehnder interferometer.

shape = 300  # Grid size
waist_radius = 500e-6  # Beam waist radius (m)

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Input Field
# -----------
# A Gaussian beam is used as the input field.

input_field = Field(gaussian(shape, waist_radius)).normalize()
input_field.visualize(title="Input Gaussian Beam")

# %%
# 50:50 Beam Splitter
# --------------------
# A 50:50 beam splitter splits the input field into two equal-amplitude copies.
# The transfer matrix is characterized by :math:`\theta = \pi/4` (equal splitting).

bs = BeamSplitter(shape, theta=torch.pi / 4, phi_0=0, phi_r=0, phi_t=0, z=0)
print("Beam splitter transfer matrix:")
print(bs.transfer_matrix)

# Split the input field
arm1, arm2 = bs(input_field)
print(f"Arm 1 power: {arm1.power().item():.4f}")
print(f"Arm 2 power: {arm2.power().item():.4f}")

# %%
# Phase Scanning Interferometry
# -----------------------------
# In a Mach-Zehnder interferometer, one arm acquires a relative phase shift.
# When the two arms are recombined at a second beam splitter, the output intensity
# depends on the phase difference. Scanning the phase produces sinusoidal fringes.

bs2 = BeamSplitter(shape, theta=torch.pi / 4, phi_0=0, phi_r=0, phi_t=0, z=0)

phase_shifts = torch.linspace(0, 2 * torch.pi, 100)
output1_powers = []
output2_powers = []

for phi in phase_shifts:
    # Apply a uniform phase shift to arm 2
    arm2_shifted = arm2.modulate(torch.exp(1j * phi))

    # Recombine at the second beam splitter
    out1, out2 = bs2(arm1, arm2_shifted)
    output1_powers.append(out1.power().item())
    output2_powers.append(out2.power().item())

# %%
# Interferometer Output vs. Phase
# --------------------------------
# The two output ports show complementary power oscillations. When one port is bright,
# the other is dark — a direct consequence of energy conservation.

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(phase_shifts.numpy(), output1_powers, label="Output Port 1", linewidth=2)
ax.plot(phase_shifts.numpy(), output2_powers, label="Output Port 2", linewidth=2)
ax.set_xlabel("Phase Shift (radians)")
ax.set_ylabel("Power (a.u.)")
ax.set_title("Mach-Zehnder Interferometer: Output Power vs. Phase Shift")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Spatial Interference Fringes
# ----------------------------
# Instead of a uniform phase shift, we can apply a linearly varying phase across
# the beam to produce spatial interference fringes at the output.

y, x = arm2.meshgrid()
fringe_period = 200e-6  # Fringe period (m)
tilt_phase = 2 * torch.pi * x / fringe_period
arm2_tilted = arm2.modulate(torch.exp(1j * tilt_phase))

# Recombine
fringed_out1, fringed_out2 = bs2(arm1, arm2_tilted)

fringed_out1.visualize(title=f"Output Port 1 — Spatial Fringes (period = {fringe_period * 1e6:.0f} μm)")
fringed_out2.visualize(title=f"Output Port 2 — Spatial Fringes (period = {fringe_period * 1e6:.0f} μm)")
