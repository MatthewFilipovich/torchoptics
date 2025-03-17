"""
Polarized Field
===============

Simulates the propagation of polarized light through different polarizers.
"""

# %%

import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import PolarizedField
from torchoptics.elements import LinearPolarizer

# %%
# Simulation Parameters
# ---------------------
# Define the grid size and set default optical properties.

shape = 100  # Grid size (number of points per dimension)

# Configure torchoptics defaults
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# Initializing a Polarized Field
# ------------------------------
# We create a uniformly polarized field along the :math:`x`-axis.

field_data = torch.zeros(3, shape, shape)
field_data[0, :, :] = 1  # Polarization along the x-axis
field = PolarizedField(field_data).normalize()

# %%
# Malus’s Law: Power Transmission Through a Rotating Polarizer
# ------------------------------------------------------------
# We apply linear polarizers at different angles ranging from 0 to :math:`2\pi` radians.
# `Malus’s law  <https://en.wikipedia.org/wiki/Polarizer#Malus's_law_and_other_properties>`_ states that the 
# transmitted power follows:
#
# .. math::
#     P = P_0 \cos^2(\theta)
#
# where :math:`P_0` is the initial power, and :math:`\theta` is the angle between the
# initial polarization and the polarizer’s axis.

angles = torch.linspace(0, 2 * torch.pi, 400)
field_power = [LinearPolarizer(shape, theta=theta)(field).power() for theta in angles]

# Plot results
plt.plot(angles, field_power)
plt.xlim(angles.min(), angles.max())
plt.ylim(0, 1)
plt.xticks(
    ticks=[0, torch.pi / 2, torch.pi, 3 * torch.pi / 2, 2 * torch.pi],
    labels=["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
)
plt.xlabel("Polarizer Angle (radians)")
plt.ylabel("Transmitted Power")
plt.title("Malus's Law: Power Transmission Through a Polarizer")
plt.grid()
plt.show()

# %%
# Sequential Polarizers and Projection Effects
# --------------------------------------------
# Applying two polarizers in sequence highlights an important property of polarization:
#
# - A 90° polarizer alone completely blocks light polarized along 0°.
# - However, inserting an intermediate 45° polarizer allows some light to pass through.
# - The second polarizer (90°) then transmits part of this newly polarized light.
#
# The first polarizer projects the field onto a new axis, enabling partial transmission through the second 
# polarizer.

polarizer_0 = LinearPolarizer(shape, theta=0)
polarizer_45 = LinearPolarizer(shape, theta=torch.pi / 4)
polarizer_90 = LinearPolarizer(shape, theta=torch.pi / 2)

# Compute power measurements
initial_power = field.power()
power_after_0 = polarizer_0(field).power()
power_after_45 = polarizer_45(field).power()
power_after_90 = polarizer_90(field).power()
power_after_45_90 = polarizer_90(polarizer_45(field)).power()

# Print results
print(f"Initial Power: {initial_power:.2f}")
print(f"Power after 0° polarizer: {power_after_0:.2f}")
print(f"Power after 45° polarizer: {power_after_45:.2f}")
print(f"Power after 90° polarizer: {power_after_90:.2f}")
print(f"Power after sequential 45° and 90° polarizers: {power_after_45_90:.2f}")
