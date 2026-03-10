.. _user-guide-systems:

Optical Systems
================

The :class:`~torchoptics.System` class models an optical system as an ordered collection of
:class:`~torchoptics.elements.Element` objects arranged along the optical axis. It automates
field propagation between elements and provides methods for measuring the field at any plane.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

A :class:`~torchoptics.System` works like :class:`torch.nn.Sequential` for optical elements: it
holds a sequence of elements and automatically propagates the field between them. The key difference
is that elements are ordered by their :math:`z`-positions rather than insertion order.

.. code-block:: python

    from torchoptics import System
    from torchoptics.elements import Lens, PhaseModulator
    import torch

    system = System(
        Lens(shape=500, focal_length=0.2, z=0.2),
        Lens(shape=500, focal_length=0.2, z=0.6),
    )
    print(system)

Creating Systems
-----------------

From Elements
^^^^^^^^^^^^^

Pass any number of :class:`~torchoptics.elements.Element` instances to the constructor:

.. code-block:: python

    from torchoptics.elements import Lens, AmplitudeModulator
    from torchoptics.profiles import circle

    system = System(
        AmplitudeModulator(circle(500, radius=2e-3), z=0.0),
        Lens(500, focal_length=0.1, z=0.1),
        Lens(500, focal_length=0.1, z=0.3),
    )

Elements can be provided in any order — the system sorts them by :math:`z`-position internally when
propagating.

Accessing Elements
^^^^^^^^^^^^^^^^^^

Systems support indexing, slicing, iteration, and length:

.. code-block:: python

    # Get elements
    first = system[0]
    last = system[-1]
    sub_system = system[1:]  # Returns a new System

    # Iterate
    for element in system:
        print(element)

    # Number of elements
    print(len(system))

    # Get sorted elements (by z-position)
    sorted_elems = system.sorted_elements()


Forward Propagation
--------------------

The ``forward()`` method (or calling the system directly) propagates a field through all elements
in order of their :math:`z`-positions:

.. code-block:: python

    output = system(input_field)

At each step, the field is:

1. **Propagated** from its current :math:`z`-position to the next element's :math:`z`-position
2. **Transformed** by the element's ``forward()`` method

The returned field is the output of the **last** element (in :math:`z`-order).

You can control the propagation method and padding:

.. code-block:: python

    output = system(input_field, propagation_method="ASM", asm_pad=2)

Measurement
------------

Often you want to know the field at a plane that doesn't correspond to any element. The measurement
methods propagate through the system and then continue to a target plane.

Measure at z
^^^^^^^^^^^^^

:meth:`~torchoptics.System.measure_at_z` propagates the field through the system and then to a
specified :math:`z`-position:

.. code-block:: python

    # Measure at the image plane (beyond the last element)
    output = system.measure_at_z(input_field, z=0.8)

    # Measure at an intermediate plane
    intermediate = system.measure_at_z(input_field, z=0.15)

The system only applies elements whose :math:`z`-positions lie between the input field's :math:`z`
and the target :math:`z`. If you measure at a :math:`z` before the last element, only elements up
to that point are applied.

Measure with Custom Geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~torchoptics.System.measure` allows specifying a full output geometry:

.. code-block:: python

    # Measure with a different grid shape and spacing
    output = system.measure(input_field, shape=300, z=0.8, spacing=20e-6)

Measure at Plane
^^^^^^^^^^^^^^^^^

:meth:`~torchoptics.System.measure_at_plane` propagates to match the geometry of a
:class:`~torchoptics.PlanarGrid`:

.. code-block:: python

    from torchoptics import PlanarGrid

    target = PlanarGrid(shape=300, z=0.8, spacing=20e-6)
    output = system.measure_at_plane(input_field, target)


Example: 4f Imaging System
----------------------------

A 4f system consists of two lenses separated by the sum of their focal lengths, with the object at one
focal length before the first lens and the image at one focal length after the second:

.. code-block:: python

    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import Lens
    from torchoptics.profiles import checkerboard

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    f = 200e-3  # Focal length
    shape = 1000

    system = System(
        Lens(shape, f, z=1 * f),
        Lens(shape, f, z=3 * f),
    )

    input_field = Field(checkerboard(shape, 400e-6, 15))

    # Measure at each focal plane
    for i in range(5):
        z = i * f
        output = system.measure_at_z(input_field, z=z)
        output.visualize(title=f"z = {i}f")

Example: Adding a Fourier Filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Fourier plane filter can be added at the midpoint of a 4f system (at :math:`z = 2f`) to perform
spatial filtering:

.. code-block:: python

    from torchoptics.elements import Modulator
    from torchoptics.profiles import circle

    # Low-pass filter: circular aperture at the Fourier plane
    filter_radius = 0.5e-3
    low_pass = Modulator(circle(shape, filter_radius), z=2 * f)

    filtered_system = System(
        Lens(shape, f, z=1 * f),
        low_pass,
        Lens(shape, f, z=3 * f),
    )

    output = filtered_system.measure_at_z(input_field, z=4 * f)
    output.visualize(title="Low-Pass Filtered Output")

GPU Acceleration
-----------------

Like PyTorch modules, systems and their elements can be moved to GPU:

.. code-block:: python

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    system = system.to(device)
    input_field = Field(checkerboard(shape, 400e-6, 15)).to(device)
    output = system.measure_at_z(input_field, z=4 * f)

All propagation and modulation computations are then performed on the GPU, which can provide
significant speedups for large grids.
