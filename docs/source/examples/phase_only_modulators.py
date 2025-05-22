"""
Shaping Fields Using Phase-Only Modulators
============================================

Demonstrates how to shape an optical field using a phase-only modulation pattern.

We follow the method described in:

    Eliot Bolduc, Nicolas Bent, Enrico Santamato, Ebrahim Karimi, and Robert W. Boyd,
    “Exact solution to simultaneous intensity and phase encryption with a single phase-only hologram,”
    Opt. Lett. 38, 3546-3549 (2013), https://doi.org/10.1364/OL.38.003546

In this example, we generate a Hermite-Gaussian beam profile using a phase-only modulator, a circular
aperture, and two lenses.
"""

# %%
import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import AmplitudeModulator, Lens, PhaseModulator
from torchoptics.profiles import circle, hermite_gaussian
from torchoptics.visualization import visualize_tensor

shape = 1000  # Grid size (number of points per dimension)
spacing = 10e-6  # Grid spacing (m)
wavelength = 700e-9  # Wavelength (m)
focal_length = 200e-3  # Lens focal length (m)

waist_radius = 0.3e-3

# Select computation device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure torchoptics defaults
torchoptics.set_default_spacing(spacing)
torchoptics.set_default_wavelength(wavelength)

# %%
# Target Field: Hermite-Gaussian Beam
# -----------------------------------
# Define the desired output field using a Hermite-Gaussian profile.

desired_field = hermite_gaussian(shape, 5, 5, waist_radius).to(device)
visualize_tensor(desired_field, title="Desired Output Field")

# %%
# Compute Phase-Only Hologram
# -------------------------------
#
# Generate a phase-only modulation pattern that reproduces the desired field
# after propagation through a lens.
#
# The hologram phase is computed as:
#
# :math:`\Psi(m, n) = \mathcal{M}(m, n) \mathrm{mod}(\mathcal{F}(m, n) + 2 \pi m / \Lambda, 2 \pi)`
#
# where :math:`\Lambda` is the period of the blazed grating,
#
# :math:`\mathcal{M} = 1 + \frac{1}{\pi} \mathrm{sinc}^{-1}(\mathcal{A})`, and
#
# :math:`\mathcal{F} = \Phi - \pi \mathcal{M}`.
#
# :math:`\mathcal{A}` is the amplitude of the desired field and :math:`\Phi` is its phase.


def create_phase_hologram(desired_field, grating_period):
    def inverse_sinc(y):
        # Approximate inverse of sinc(πx) for x ∈ [−π, 0], adapted from:
        # https://math.stackexchange.com/a/3345578
        z = 1 - y
        return -torch.sqrt(12 * z / (1 - 0.20409 * z + torch.sqrt(1 - 0.792 * z - 0.0318 * z**2)))

    angle = torch.angle(desired_field)
    amplitude = torch.abs(desired_field)
    amplitude /= torch.max(amplitude)  # Normalize amplitude to [0, 1]

    M = 1 + 1 / torch.pi * inverse_sinc(amplitude)
    F = angle - torch.pi * M
    m = torch.arange(desired_field.shape[0]) * spacing
    return M * ((F + 2 * torch.pi * m / grating_period) % (2 * torch.pi))


# %%
# Generate and visualize the phase-only hologram
hologram_phase = create_phase_hologram(desired_field, grating_period=40e-6)
phase_modulator = PhaseModulator(hologram_phase).to(device)
phase_modulator.visualize(title="Phase-Only Hologram")

# %%
# 2f Beam Shaping System
# ----------------------
# Construct a 2f system: phase modulator → lens → focal plane (2f).

system = System(phase_modulator, Lens(shape, focal_length, z=focal_length)).to(device)

# Input is a uniform plane wave
input_field = Field(torch.ones(shape, shape)).to(device)

# Measure the output field at z = 2f
output_field = system.measure_at_z(input_field, z=2 * focal_length)
output_field.visualize(title="Output Field at 2f", vmax=1)

# %%
# Circular Aperture at 2f Plane
# -----------------------------
# Insert a circular amplitude mask at the focal plane (2f) to spatially filter the output.

circular_aperture = AmplitudeModulator(circle(shape, radius=1e-3, offset=(0, 3.5e-3)), z=2 * focal_length).to(
    device
)

circular_aperture.visualize(title="Circular Aperture at 2f")

# %%
# 4f System: Modulator → Lens → Aperture → Lens
# ---------------------------------------------
# Extend the system to include an aperture and a second lens.
# Measure the field at the output plane (4f).

extended_system = System(
    phase_modulator,
    Lens(shape, focal_length, z=focal_length),
    circular_aperture,
    Lens(shape, focal_length, z=3 * focal_length),
).to(device)

output_field = extended_system.measure_at_z(input_field, z=4 * focal_length)
output_field.visualize(title="Final Output Field at 4f")
