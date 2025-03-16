"""
1D Simulations
===============

Simulates 1D field propagation using the angular spectrum method (ASM).

We can perform 1D simulations using ASM along the :math:`x`-axis by ensuring the ``shape`` and
``asm_pad_factor`` along the :math:`y`-axis are set to ``1`` and ``0``, respectively.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field
from torchoptics.profiles import square

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

field = Field(square((1000, 1), side=200e-5))
x = field.meshgrid()[0]

plt.plot(x, field.intensity().squeeze())
plt.title("Input field")
plt.xlabel("x (m)")
plt.ylabel("Intensity")
plt.xlim(x.min(), x.max())
plt.ylim(0, 2)
plt.show()

# %%
# Let's propagate the field using ASM. We can visualize the intensity at different propagation distances.
# We will set the ``asm_pad_factor`` to ``(50, 0)``, which effectively only performs ASM along the
# :math:`x`-axis.

for z in [0, 0.5, 1, 1.5, 2, 2.5]:
    fig, ax = plt.subplots()
    propagated_field = field.propagate_to_z(z, asm_pad_factor=(50, 0), propagation_method="asm")
    ax.plot(x, propagated_field.intensity().squeeze())
    ax.set_title(f"z = {z} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 2)
# %%
# We can also animate the propagation of the field using matplotlib.

num_frames = 301
z_list = torch.linspace(0, 3, num_frames)

intensities = [
    field.propagate_to_z(z, asm_pad_factor=(50, 0), propagation_method="asm").intensity().squeeze()
    for z in z_list
]

fig, ax = plt.subplots()
line2 = ax.plot(x, intensities[0])[0]
ax.set(xlim=[x.min(), x.max()], ylim=[0, 2], xlabel="x [m]", ylabel="Intensity")
ax.set_xlabel("x (m)")
ax.set_ylabel("Intensity")


def update(frame):
    line2.set_ydata(intensities[frame])
    ax.set_title(f"z = {z_list[frame]:.2f} m")
    return line2


ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=30)
plt.show()
