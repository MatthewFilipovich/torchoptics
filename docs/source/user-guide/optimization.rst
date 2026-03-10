.. _user-guide-optimization:

Optimization & Training
=========================

One of the most powerful features of TorchOptics is its full differentiability — every operation
from field propagation to modulation and detection supports automatic differentiation via PyTorch's
autograd. This enables gradient-based optimization of optical system parameters.

.. contents:: On This Page
   :local:
   :depth: 2

Why Differentiable Optics?
----------------------------

Traditional optical design relies on ray tracing and iterative manual adjustment. Differentiable
wave optics enables:

- **Inverse design**: Directly optimize optical element parameters to achieve a target output
- **End-to-end training**: Jointly optimize optical hardware and downstream machine learning models
- **Diffractive optical elements**: Design phase masks that perform computation with light
- **Rapid prototyping**: Explore large design spaces using gradient descent

Making Parameters Trainable
-----------------------------

Any element parameter can be made trainable by wrapping it with :class:`torch.nn.Parameter`:

.. code-block:: python

    import torch
    from torch.nn import Parameter
    from torchoptics.elements import PhaseModulator, Lens

    # Trainable phase modulator (e.g., spatial light modulator)
    phase = Parameter(torch.zeros(200, 200))
    slm = PhaseModulator(phase, z=0.1)

    # Trainable lens with learnable focal length
    f = Parameter(torch.tensor(0.2))
    lens = Lens(shape=200, focal_length=f, z=0.3)

Parameters wrapped in :class:`~torch.nn.Parameter` are automatically tracked by PyTorch and
updated during optimization.

Defining a Loss Function
--------------------------

The loss function measures how well the current optical system performs. Common choices include:

**Mean Squared Error** — Minimize the difference between output and target intensity:

.. code-block:: python

    def mse_loss(output_field, target_field):
        return torch.mean((output_field.intensity() - target_field.intensity()) ** 2)

**Overlap Integral** — Maximize the overlap between output and target fields:

.. code-block:: python

    def overlap_loss(output_field, target_field):
        overlap = output_field.inner(target_field).abs() ** 2
        return -overlap  # Negative because we minimize

**Power in Target Region** — Maximize power in a desired region:

.. code-block:: python

    def efficiency_loss(output_field, target_mask):
        power_in_target = (output_field.intensity() * target_mask).sum()
        total_power = output_field.power()
        return -power_in_target / total_power  # Maximize efficiency


Training Loop
--------------

A typical optimization follows the standard PyTorch training pattern:

.. code-block:: python

    import torch
    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import PhaseModulator
    from torchoptics.profiles import gaussian

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    shape = 200

    # Input: Gaussian beam
    input_field = Field(gaussian(shape, waist_radius=300e-6))

    # Target: desired output intensity
    target_field = Field(gaussian(shape, waist_radius=100e-6), z=0.5)

    # System: trainable phase modulator
    system = System(
        PhaseModulator(torch.nn.Parameter(torch.zeros(shape, shape)), z=0.0),
    )

    # Optimizer
    optimizer = torch.optim.Adam(system.parameters(), lr=0.1)

    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()

        # Forward pass: propagate through the system
        output = system.measure_at_z(input_field, z=0.5)

        # Compute loss
        loss = torch.mean(
            (output.intensity() - target_field.intensity()) ** 2
        )

        # Backward pass: compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")


Example: Diffractive Beam Splitter
-------------------------------------

This example trains a diffractive optical system to split a single Gaussian beam into four spots:

.. code-block:: python

    import torch
    from torch.nn import Parameter
    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import PhaseModulator
    from torchoptics.profiles import gaussian

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    shape = 250
    waist_radius = 150e-6
    offset = 3.8 * waist_radius

    # Input field: single Gaussian beam
    input_field = Field(gaussian(shape, waist_radius))

    # Target: four Gaussian spots
    target_data = (
        gaussian(shape, waist_radius, offset=(offset, offset))
        + gaussian(shape, waist_radius, offset=(offset, -offset))
        + gaussian(shape, waist_radius, offset=(-offset, offset))
        + gaussian(shape, waist_radius, offset=(-offset, -offset))
    ) / 2
    target_field = Field(target_data, z=0.6)

    # Three-layer diffractive system
    system = System(
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.0),
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.2),
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.4),
    )

    # Train
    optimizer = torch.optim.Adam(system.parameters(), lr=0.1)

    for epoch in range(200):
        optimizer.zero_grad()
        output = system.measure_at_z(input_field, z=0.6)
        loss = -output.inner(target_field).abs() ** 2
        loss.backward()
        optimizer.step()

GPU Acceleration
-----------------

For large grids, move everything to GPU for significant speedups:

.. code-block:: python

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_field = input_field.to(device)
    target_field = target_field.to(device)
    system = system.to(device)

    # Training proceeds as before — all computations on GPU

Tips for Effective Optimization
---------------------------------

Learning Rate
^^^^^^^^^^^^^

- Phase modulators often work well with learning rates of 0.01 to 0.1
- Geometric parameters (focal length, positions) may need smaller rates (1e-4 to 1e-3)
- Use learning rate schedulers for fine-tuning:

.. code-block:: python

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(200):
        # ... training step ...
        scheduler.step()

Optimizer Choice
^^^^^^^^^^^^^^^^^

- **Adam** is generally a good default for optical design problems
- **SGD with momentum** can sometimes find better solutions for simple systems
- **L-BFGS** can be very effective for small parameter counts but requires more memory

Multiple Phase Layers
^^^^^^^^^^^^^^^^^^^^^^

For complex transformations, use multiple phase modulator layers separated in :math:`z`. This
provides more degrees of freedom:

.. code-block:: python

    system = System(
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.0),
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.1),
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.2),
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.3),
    )

Regularization
^^^^^^^^^^^^^^^

To encourage smoother phase profiles or manufacturing-feasible designs:

.. code-block:: python

    # Total variation regularization on the phase
    phase = system[0].phase
    tv = torch.mean(torch.abs(phase[:, :-1] - phase[:, 1:])) + \
         torch.mean(torch.abs(phase[:-1, :] - phase[1:, :]))
    total_loss = loss + 0.01 * tv

Integrating with Neural Networks
----------------------------------

Since TorchOptics elements are :class:`torch.nn.Module` subclasses, they integrate seamlessly
with PyTorch's ecosystem. You can combine optical systems with neural networks for hybrid
optical-digital processing:

.. code-block:: python

    import torch.nn as nn

    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.optical_system = System(
                PhaseModulator(Parameter(torch.zeros(200, 200)), z=0.0),
                Lens(200, focal_length=0.1, z=0.1),
            )
            self.digital_network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(200 * 200, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )

        def forward(self, input_field):
            # Optical processing
            output_field = self.optical_system.measure_at_z(input_field, z=0.2)
            intensity = output_field.intensity()

            # Digital processing
            return self.digital_network(intensity)
