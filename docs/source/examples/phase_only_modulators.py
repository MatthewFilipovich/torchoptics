"""
Phase Only Modulators
=====================

Simulates holography using phase-only modulators.
"""

# %%
import torch

import torchoptics
from torchoptics import Field, System
from torchoptics.elements import AmplitudeModulator, Lens, PhaseModulator
from torchoptics.profiles import blazed_grating, circle, hermite_gaussian
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
# Create the Desired Output Field using a Hermite-Gaussian Profile

desired_field = hermite_gaussian(shape, 5, 5, waist_radius).to(device)
visualize_tensor(desired_field, title="Desired Field")


# %%
# Compute a Phase-Only Hologram to Reproduce the Desired Output Field


def create_phase_hologram(desired_field, grating_period):
    # Exact solution to simultaneous intensity and phase encoding with a single phase-only hologram.
    # https://doi.org/10.1364/OL.38.003546

    def inverse_sinc(x):
        z = 1 - x
        return torch.sqrt((12 * z) / (1 - (0.20409 * z) + torch.sqrt(1 - (0.792 * z) - (0.0318 * z * z))))

    angle = torch.angle(desired_field)
    amplitude = torch.abs(desired_field)
    amplitude /= torch.max(amplitude)  # Normalize amplitude to [0, 1]

    grating = blazed_grating(desired_field.shape, grating_period, height=2 * torch.pi)
    phi = (angle + grating) % (2 * torch.pi)

    scaling = 1 - 1 / torch.pi * inverse_sinc(amplitude)
    hologram = scaling * ((phi - torch.pi * scaling) % (2 * torch.pi))

    return hologram


hologram_phase = create_phase_hologram(desired_field, 40e-6)
phase_modulator = PhaseModulator(hologram_phase).to(device)
phase_modulator.visualize(title="Phase Modulator Hologram")

# %%
# Construct the Optical System with a Phase Modulator and Lens


system = System(phase_modulator, Lens(shape, focal_length, z=focal_length)).to(device)
print(system)
# %%
# Measure the Field at the Focal Plane (2f) After Propagation Through the System

input_field = Field(torch.ones(shape, shape)).to(device)
output_field = system.measure_at_z(input_field, z=2 * focal_length)
output_field.visualize(title="2f", vmax=1)

# %%
# Insert a Circular Amplitude Aperture at the Focal Plane to Modify the Output

circular_aperture = AmplitudeModulator(circle(shape, 1e-3, offset=(-3.5e-3, 0)), z=2 * focal_length).to(
    device
)
circular_aperture.visualize(title="Circular Aperture at 2f")

# %%
updated_system = System(
    phase_modulator,
    Lens(shape, focal_length, z=focal_length),
    circular_aperture,
    Lens(shape, focal_length, z=3 * focal_length),
).to(device)
print(system)

output_field = updated_system.measure_at_z(input_field, z=4 * focal_length)
output_field.visualize(title="Output Field (4f)")
