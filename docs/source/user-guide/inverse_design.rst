Inverse Design
==============


Every TorchOptics operation (propagation, modulation, detection) is fully differentiable
through :mod:`torch.autograd`. This enables gradient-based optimization of optical systems using
the same tools and patterns used to train neural networks.


Trainable Properties
--------------------

Wrap any property value in :class:`torch.nn.Parameter` to make it learnable:

**Modulation data** — the most common case:

.. code-block:: python

    import torch
    from torch.nn import Parameter
    from torchoptics.elements import PhaseModulator

    slm = PhaseModulator(Parameter(torch.zeros(300, 300)), z=0)
    list(slm.parameters())  # [Parameter containing: 300×300 tensor]

**Scalar properties** — focal length, position, angle, etc.:

.. code-block:: python

    from torchoptics.elements import Lens, LinearPolarizer

    lens = Lens(300, focal_length=Parameter(torch.tensor(100e-3)), z=0)
    pol = LinearPolarizer(300, theta=Parameter(torch.tensor(0.0)), z=0)

This works for every registered property (``z``, ``spacing``, ``offset``, ``focal_length``,
``theta``, ``phi``, etc.). If the value is a ``Parameter``, it is learnable; otherwise it is a
fixed buffer.


Parameterization
----------------

For unconstrained phase modulation, ``PhaseModulator(Parameter(torch.zeros(...)))`` already
works; gradients flow directly through the phase values and the optimizer is free to explore
all real numbers.

When you need to **constrain** a parameter to a physical range, use
:func:`torch.nn.utils.parametrize.register_parametrization`. For example, amplitude must
stay in :math:`[0, 1]`; register a sigmoid parametrization directly on the
``AmplitudeModulator``'s ``amplitude`` parameter:

.. code-block:: python

    import torch
    import torch.nn.utils.parametrize as parametrize
    from torch.nn import Parameter
    from torchoptics.elements import AmplitudeModulator

    slm = AmplitudeModulator(Parameter(torch.zeros(300, 300)), z=0)
    parametrize.register_parametrization(slm, "amplitude", torch.nn.Sigmoid())
    # slm.amplitude is always sigmoid(raw) ∈ (0, 1); the optimizer trains the raw logits

The same pattern works for any differentiable constraint; use ``torch.nn.Softplus()`` for
positive-only values, or write a custom :class:`torch.nn.Module` for arbitrary mappings.


Training Loop
-------------

The standard PyTorch training loop applies directly:

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
    input_field = Field(gaussian(shape, waist_radius=300e-6), z=0)

    system = System(
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.0),
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.2),
    )

    optimizer = torch.optim.Adam(system.parameters(), lr=0.1)

    for iteration in range(200):
        optimizer.zero_grad()
        output = system.measure_at_z(input_field, z=0.4)
        loss = compute_loss(output)
        loss.backward()
        optimizer.step()

The :meth:`~torchoptics.Field.inner` method provides a natural loss for mode matching:

.. code-block:: python

    overlap = output.inner(target_field).abs().square()
    loss = 1 - overlap

See the :doc:`/quickstart/index` for a complete end-to-end training example.


Saving and Loading
------------------

Use PyTorch's standard serialization:

.. code-block:: python

    torch.save(system.state_dict(), "optimized_system.pt")
    system.load_state_dict(torch.load("optimized_system.pt"))


GPU Acceleration
----------------

For large grids and long training loops, move everything to the GPU for significant speedups.
See :doc:`configuration` for device management.
