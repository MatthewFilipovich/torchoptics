"""
Laser Speckle Patterns
======================

Simulates laser speckle, the granular intensity pattern formed when coherent
light scatters from a rough surface or propagates through a random medium.
Speckle is ubiquitous in laser applications and has both practical uses
(speckle imaging, metrology) and challenges (image degradation).
"""

# %%

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field, visualize_tensor
from torchoptics.profiles import circle, gaussian

# %%
# Simulation Parameters
# ---------------------
# Speckle arises from the interference of many wavefronts with random phases.
# The characteristic speckle size depends on the wavelength, propagation
# distance, and the size of the illuminated area.

shape = 512  # Grid size
spacing = 5e-6  # Grid spacing (m)
wavelength = 632.8e-9  # HeNe laser (m)

beam_radius = 1e-3  # Illumination beam radius (m)

# Configure torchoptics defaults
torchoptics.set_default_spacing(spacing)
torchoptics.set_default_wavelength(wavelength)

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Random Phase Screen
# -------------------
# A rough surface introduces random phase variations. The phase is often
# modeled as uniformly distributed over :math:`[0, 2\pi]`, representing
# surface height variations much larger than the wavelength.

# Create random phase screen (fully developed speckle)
random_phase = 2 * torch.pi * torch.rand(shape, shape)

# Illumination: Gaussian beam
amplitude = gaussian(shape, waist_radius=beam_radius)

# Create Field object
input_field = Field(amplitude * torch.exp(1j * random_phase)).to(device)

input_field.visualize(title="Input Field")

# %%
# Far-Field Speckle
# -----------------
# After propagation, the random phases interfere to create the characteristic
# granular speckle pattern. The speckle size in the far field is approximately:
#
# .. math::
#     \delta_s \approx \frac{\lambda z}{D}
#
# where :math:`D` is the diameter of the illuminated area.

# Propagate to far field
propagation_distance = 100e-3  # 100 mm
output_field = input_field.propagate_to_z(propagation_distance)
speckle_pattern = output_field.intensity().cpu()

visualize_tensor(speckle_pattern, title="Far-Field Laser Speckle", cmap="hot")

# %%
# Speckle Statistics
# ------------------
# For fully developed speckle (uniform random phase), the intensity follows
# a negative exponential distribution:
#
# .. math::
#     p(I) = \frac{1}{\langle I \rangle} \exp\left(-\frac{I}{\langle I \rangle}\right)
#
# The standard deviation equals the mean, giving a speckle contrast of 1.

# Flatten the speckle pattern for statistics
speckle_flat = speckle_pattern.flatten()
mean_intensity = speckle_flat.mean()
std_intensity = speckle_flat.std()
contrast = std_intensity / mean_intensity

print(f"Mean intensity: {mean_intensity:.4f}")
print(f"Standard deviation: {std_intensity:.4f}")
print(f"Speckle contrast (C = σ/μ): {contrast:.3f}")
print("Theoretical contrast for fully developed speckle: 1.0")

# Histogram
fig, ax = plt.subplots(figsize=(8, 5))

# Normalize intensities
normalized = speckle_flat / mean_intensity
ax.hist(normalized, bins=100, density=True, alpha=0.7, color="steelblue", label="Simulation")

# Theoretical negative exponential
I_theory = torch.linspace(0, 8, 200)
p_theory = torch.exp(-I_theory)
ax.plot(I_theory, p_theory, "r-", linewidth=2, label=r"Theory: $p(I) = e^{-I/\langle I \rangle}$")

ax.set_xlabel(r"Normalized Intensity $I/\langle I \rangle$")
ax.set_ylabel("Probability Density")
ax.set_title("Speckle Intensity Distribution")
ax.legend()
ax.set_xlim(0, 8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Speckle Size vs. Aperture
# -------------------------
# The average speckle size increases as the illuminated area decreases.
# We demonstrate this with different aperture sizes.

aperture_radii = [2e-3, 1e-3, 0.5e-3]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

for ax, radius in zip(axes, aperture_radii):
    # Circular aperture
    aperture = circle(shape, radius)
    scattered = aperture * torch.exp(1j * random_phase)
    field = Field(scattered).to(device)
    output = field.propagate_to_z(propagation_distance)

    ax.imshow(output.intensity().cpu(), cmap="hot")
    ax.set_title(f"Aperture r = {radius * 1e3:.1f} mm")
    ax.axis("off")

plt.suptitle("Speckle Size vs. Aperture Size", fontsize=12)
plt.show()
