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

# Set simulation properties
shape = 100
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# Initialize a polarized field
field_data = torch.zeros(3, shape, shape)
field_data[0, :, :] = 1  # Polarization along the x-axis
field = PolarizedField(field_data).normalize()

# %%
# Linear Polarizers with Varying Orientation
# -------------------------------------------
# We apply a series of linear polarizers at different angles, ranging from 0 to :math:`2\pi` radians,
# and measure the transmitted power. According to `Malus's law
# <https://en.wikipedia.org/wiki/Polarizer#Malus's_law_and_other_properties>`_, the power after passing
# through a polarizer is given by:
#
# .. math::
#     P = P_0 \cos^2(\theta)
#
# where :math:`P_0` is the initial power of the light, and :math:`\theta` is the angle between the
# polarization direction and the polarizer's transmission axis.

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
# Sequential Polarizers
# ----------------------
# Applying two polarizers in sequence highlights an important property of polarization:
#
# - A single polarizer at 90° blocks light that starts polarized along 0°.
# - However, inserting an intermediate polarizer at 45° allows some light to pass through.
# - This happens because the first polarizer projects the field onto a new axis,
#   allowing a component to survive the second projection.

polarizer_0 = LinearPolarizer(shape, theta=0)
polarizer_45 = LinearPolarizer(shape, theta=torch.pi / 4)
polarizer_90 = LinearPolarizer(shape, theta=torch.pi / 2)

print("Initial Power:", field.power())
print("Power after 0° polarizer:", polarizer_0(field).power())
print("Power after 45° polarizer:", polarizer_45(field).power())
print("Power after 90° polarizer:", polarizer_90(field).power())
print("Power after sequential 45° and 90° polarizers:", polarizer_90(polarizer_45(field)).power())
