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

# Set default simulation parameters
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# Initialize a 1D field
field = Field(square((1000, 1), side=200e-5))
x = field.meshgrid()[0]

# Plot the input field
plt.plot(x, field.intensity().squeeze())
plt.title("Input Field")
plt.xlabel("x (m)")
plt.ylabel("Intensity")
plt.xlim(x.min(), x.max())
plt.ylim(0, 2)
plt.show()

# %%
# Propagation Using ASM
# ---------------------
# We propagate the field using the angular spectrum method (ASM) and visualize
# its intensity at different distances. The ``asm_pad_factor`` is set to ``(50, 0)``,
# ensuring propagation occurs only along the :math:`x`-axis.

z_values = [0, 0.5, 1, 1.5, 2, 2.5]
fig, axes = plt.subplots(len(z_values), 1, figsize=(6, 15), sharex=True)

for ax, z in zip(axes, z_values):
    propagated_field = field.propagate_to_z(z, asm_pad_factor=(50, 0), propagation_method="asm")
    ax.plot(x, propagated_field.intensity().squeeze())
    ax.set_title(f"z = {z} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 2)

plt.tight_layout()
plt.show()

# %%
# Animated Propagation
# --------------------
# We animate the field propagation over a continuous range of distances.

num_frames = 301
z_list = torch.linspace(0, 3, num_frames)

# Compute intensities at each propagation step
intensities = [
    field.propagate_to_z(z, asm_pad_factor=(50, 0), propagation_method="asm").intensity().squeeze()
    for z in z_list
]

# Create figure and axis for animation
fig, ax = plt.subplots()
(line,) = ax.plot(x, intensities[0])
ax.set(xlim=[x.min(), x.max()], ylim=[0, 2], xlabel="x (m)", ylabel="Intensity")


def update(frame):
    """Update function for the animation."""
    line.set_ydata(intensities[frame])
    ax.set_title(f"z = {z_list[frame]:.2f} m")
    return (line,)


# Generate and display the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=30)
plt.show()
