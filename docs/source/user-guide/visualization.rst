.. _user-guide-visualization:

Visualization
==============

TorchOptics provides built-in visualization tools for inspecting optical fields, element profiles,
and simulation results using Matplotlib.

.. contents:: On This Page
   :local:
   :depth: 2

Field Visualization
--------------------

The :meth:`~torchoptics.Field.visualize` method is the primary way to inspect fields:

.. code-block:: python

    field.visualize(title="My Field")

For **complex fields**, this displays two subplots:

- **Left**: Squared magnitude (intensity) :math:`|U|^2`
- **Right**: Phase :math:`\arg(U)` in the range :math:`[-\pi, \pi]`

For **real fields**, a single plot is shown.

Customizing Field Plots
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``visualize()`` method accepts keyword arguments passed to Matplotlib's ``imshow()``:

.. code-block:: python

    # Set intensity limits
    field.visualize(vmin=0, vmax=1, title="Clamped Intensity")

    # Custom colormap
    field.visualize(cmap="hot", title="Hot Colormap")

Intensity-Only Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To show only the intensity (no phase subplot):

.. code-block:: python

    field.visualize(intensity=True, title="Intensity Only")

Indexing into Batch/Polarized Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For fields with extra dimensions (batch or polarization), pass index arguments to select a slice:

.. code-block:: python

    # Batch of 10 fields
    batch_field.visualize(3, title="Fourth field in batch")

    # Polarized field — x-component
    polarized_field.visualize(0, title="Ex component")

    # Batch of polarized fields — 5th batch, y-component
    field.visualize(4, 1, title="Batch 5, Ey")


Element Visualization
----------------------

Optical elements have their own ``visualize()`` method that displays their modulation profile:

.. code-block:: python

    from torchoptics.elements import Lens

    lens = Lens(shape=500, focal_length=0.2, z=0.3)
    lens.visualize(title="Lens Profile")

For modulation elements, this shows the complex modulation profile (magnitude and phase).
For detectors, it shows the weight matrix.

Linear Detector Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.LinearDetector` has a specialized visualization:

.. code-block:: python

    from torchoptics.elements import LinearDetector

    detector = LinearDetector(weight, z=1.0)

    # Show individual weight channels
    detector.visualize(0, title="Channel 0 Weights")

    # Show summed weight pattern
    detector.visualize(sum_weight=True, title="Total Weight")


Standalone Visualization Functions
------------------------------------

visualize_tensor
^^^^^^^^^^^^^^^^^

:func:`~torchoptics.visualize_tensor` displays any 2D tensor, applying the same
complex/real logic as field visualization:

.. code-block:: python

    from torchoptics import visualize_tensor

    # Visualize a complex tensor
    visualize_tensor(complex_tensor, title="Complex Data")

    # Visualize a real tensor with custom labels
    visualize_tensor(
        real_tensor,
        title="Intensity Distribution",
        xlabel="x (mm)",
        ylabel="y (mm)",
    )

    # Get the figure object instead of showing
    fig = visualize_tensor(tensor, return_fig=True, show=False)

animate_tensor
^^^^^^^^^^^^^^^

:func:`~torchoptics.animate_tensor` creates animations from 3D tensors, where the first
dimension is treated as the time/frame axis:

.. code-block:: python

    from torchoptics import animate_tensor

    # Create a 3D tensor: (frames, height, width)
    frames = torch.stack([
        field.propagate_to_z(z).intensity()
        for z in torch.linspace(0, 0.5, 50)
    ])

    # Create an animation
    anim = animate_tensor(frames, title="Propagation Animation")

The animation can be saved or displayed inline in Jupyter notebooks.


Matplotlib Integration
-----------------------

Since TorchOptics uses Matplotlib under the hood, you can combine TorchOptics visualizations
with custom Matplotlib plots:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot intensities at different z-positions
    for ax, z, label in zip(axes, [0, 0.1, 0.2], ["z=0", "z=0.1m", "z=0.2m"]):
        prop = field.propagate_to_z(z)
        ax.imshow(prop.intensity().numpy(), cmap="inferno")
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

You can also retrieve figure objects from the visualization functions:

.. code-block:: python

    fig = field.visualize(title="My Field")  # Returns Figure when show=True
