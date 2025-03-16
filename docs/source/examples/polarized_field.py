"""
Polarized Field
================

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

# Initialize polarized field
field_data = torch.zeros(3, shape, shape)
field_data[0, :, :] = 1  # Polarized along x-axis
field = PolarizedField(field_data).normalize()

# %%
# Simulate the propagation of the polarized field thorugh linear polarizers with angle :math:`\theta`, which 
# are rotated from 0 to :math:`2\pi` radians. This is given by `Malus's law 
# <https://en.wikipedia.org/wiki/Polarizer#Malus's_law_and_other_properties>`_, which states that the power 
# of the light after passing through a polarizer is given by:
# 
# .. math::
#     P = P_0 \cos^2(\theta)
#
# where :math:`P_0` is the initial power of the light and :math:`\theta` is the angle between the light's 
# polarization direction and the polarizer's transmission axis.
field_power = []
angles = torch.linspace(0, 2 * torch.pi, 400)

for theta in angles:
    polarizer = LinearPolarizer(shape, theta=theta)
    field_power.append(polarizer(field).power())

plt.plot(angles, field_power)   
plt.xlim(angles.min(), angles.max())
plt.ylim(0, 1)
plt.xticks(
    ticks=[0, torch.pi / 2, torch.pi, 3 * torch.pi / 2, 2 * torch.pi],
    labels=["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
)
plt.xlabel("Angle (radians)")
plt.ylabel("Power")
plt.show()


# %%
# Let's test the polarizer with some specific angles:
polarizer_0 = LinearPolarizer(shape, theta=0)
polarizer_45 = LinearPolarizer(shape, theta=torch.pi / 4)
polarizer_90 = LinearPolarizer(shape, theta=torch.pi / 2)

print(field.power())  # 1
print(polarizer_0(field).power())  # 1
print(polarizer_45(field).power())  # 0.5
print(polarizer_90(field).power())  # 0
print(polarizer_90(polarizer_45(field)).power())  # 0.25
