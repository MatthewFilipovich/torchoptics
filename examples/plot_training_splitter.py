"""
Training Diffractive Splitter
==============================

This example demonstrates how to train a diffractive optical system to create a desired output field.
"""

# %%
import torch
import torchoptics
import matplotlib.pyplot as plt
from torch.nn import Parameter

from torchoptics import Field, System
from torchoptics.elements import PhaseModulator
from torchoptics.profiles import gaussian

# %%
shape = 250
waist_radius = 150e-6
device = "cuda" if torch.cuda.is_available() else "cpu"

torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# %%
target_field_data = (
    gaussian(shape, waist_radius, offset=(3.8 * waist_radius, 3.8 * waist_radius))
    + gaussian(shape, waist_radius, offset=(3.8 * waist_radius, -3.8 * waist_radius))
    + gaussian(shape, waist_radius, offset=(-3.8 * waist_radius, 3.8 * waist_radius))
    + gaussian(shape, waist_radius, offset=(-3.8 * waist_radius, -3.8 * waist_radius))
) / 2
target_field = Field(target_field_data, z=0.6).to(device)

target_field.visualize(title="Target field")

# %%
input_field = Field(gaussian(shape, waist_radius), z=0).to(device)
input_field.visualize(title="Input field")

# %%
system = System(
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.0),
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.2),
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.4),
).to(device)

# %%
optimizer = torch.optim.Adam(system.parameters(), lr=0.1)
losses = []
for iteration in range(20):
    optimizer.zero_grad()
    output_field = system.measure_at_z(input_field, 0.6)
    loss = 1 - output_field.inner(target_field).abs().square()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss {loss.item()}")

# %%
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss vs iteration")
plt.yscale("log")
plt.xlim(0, len(losses))
plt.show()

# %%
for i, element in enumerate(system):
    element.visualize(title=f"Modulator {i+1}")

# %%
output_field = system.measure_at_z(input_field, 0.6)
output_field.visualize(title="Output field")
