"""
Waveplates and Circular Polarization
======================================

Demonstrates the use of waveplates to convert between linear and circular polarization states.
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field
from torchoptics.elements import HalfWaveplate, QuarterWaveplate

# %%
# Simulation Parameters
# ---------------------
# We define a simple grid and set default spacing and wavelength.

shape = 100  # Grid size

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
# X-Polarized Input Field
# -----------------------
# We create a uniformly polarized field with all energy in the :math:`x`-component.

field_data = torch.zeros(3, shape, shape)
field_data[0] = 1  # E_x = 1
input_field = Field(field_data).normalize()

print("Input field power:", input_field.power().sum().item())

# %%
# Quarter-Wave Plate: Linear to Circular Polarization
# ----------------------------------------------------
# A quarter-wave plate (QWP) with its fast axis at 45° converts linearly polarized
# light to circularly polarized light. The :math:`x` and :math:`y` components become
# equal in amplitude with a :math:`\pi/2` phase difference.

qwp = QuarterWaveplate(shape, theta=torch.pi / 4, z=0)
circular_field = qwp(input_field)

# Split into polarization components
ex, ey, ez = circular_field.polarized_split()
print(f"After QWP — Ex power: {ex.power().sum().item():.4f}, Ey power: {ey.power().sum().item():.4f}")

# %%
# Verify Circular Polarization
# ----------------------------
# Circularly polarized light should have equal Ex and Ey power. We also verify
# the phase difference between the two components.

ex_phase = torch.angle(ex.data[0, shape // 2, shape // 2])
ey_phase = torch.angle(ey.data[1, shape // 2, shape // 2])
phase_diff = (ey_phase - ex_phase).item()
print(f"Phase difference (Ey - Ex): {phase_diff:.4f} rad (expected: ±π/2 ≈ ±{torch.pi / 2:.4f})")

# %%
# Half-Wave Plate: Rotating Polarization
# ----------------------------------------
# A half-wave plate (HWP) rotates the polarization direction. With the fast axis
# at angle :math:`\theta`, x-polarized light is rotated by :math:`2\theta`.

angles = torch.linspace(0, torch.pi / 2, 5)
fig, axes = plt.subplots(1, len(angles), figsize=(15, 3))

for ax, theta in zip(axes, angles):
    hwp = HalfWaveplate(shape, theta=theta, z=0)
    output = hwp(input_field)
    ex, ey, ez = output.polarized_split()
    ex_power = ex.power().sum().item()
    ey_power = ey.power().sum().item()
    ax.bar(["Ex", "Ey"], [ex_power, ey_power], color=["tab:blue", "tab:orange"])
    ax.set_title(f"θ = {theta.item() * 180 / torch.pi:.0f}°")
    ax.set_ylim(0, 1.1)

fig.suptitle("Polarization After Half-Wave Plate at Different Angles")
plt.tight_layout()
plt.show()

# %%
# QWP + HWP: Arbitrary Polarization Control
# -------------------------------------------
# Combining a QWP and HWP allows full control over the polarization state.
# Here we use a HWP to rotate x-polarization to 45°, then a QWP to convert to circular.

hwp = HalfWaveplate(shape, theta=torch.pi / 8, z=0)  # Rotate by 2×22.5° = 45°
rotated = hwp(input_field)
ex, ey, ez = rotated.polarized_split()
print(f"After HWP(22.5°) — Ex power: {ex.power().sum().item():.4f}, Ey power: {ey.power().sum().item():.4f}")
