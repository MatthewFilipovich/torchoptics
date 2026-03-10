.. _user-guide-fields:

Optical Fields
===============

The :class:`~torchoptics.Field` class is the central data structure in TorchOptics. It represents
a monochromatic complex-valued optical field sampled on a 2D planar grid perpendicular to the
optical axis.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

An optical field :math:`U(x, y)` is a complex-valued function that describes the amplitude and phase
of a monochromatic wave at every point in a plane. In TorchOptics, this continuous field is discretized
on a rectangular grid with specified spacing and shape.

Each :class:`~torchoptics.Field` stores:

- **data**: A complex-valued :class:`torch.Tensor` of shape ``(..., H, W)``
- **wavelength**: The wavelength :math:`\lambda` of the monochromatic light
- **z**: The position along the optical axis
- **spacing**: The distance between adjacent grid points ``(dy, dx)``
- **offset**: The center coordinates ``(y0, x0)`` of the grid

Creating Fields
---------------

From a Tensor
^^^^^^^^^^^^^

The most direct way to create a field is from a 2D (or higher-dimensional) tensor:

.. code-block:: python

    import torch
    from torchoptics import Field

    # Create a simple 100×100 field
    data = torch.ones(100, 100, dtype=torch.complex64)
    field = Field(data, wavelength=700e-9, spacing=10e-6)

If global defaults are set (see :ref:`user-guide-configuration`), you can omit ``wavelength`` and
``spacing``:

.. code-block:: python

    import torchoptics

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    field = Field(torch.ones(100, 100))  # Uses defaults

.. note::
    Real-valued input tensors are automatically converted to complex type. You do not need to explicitly
    cast to complex dtype.

From Profile Functions
^^^^^^^^^^^^^^^^^^^^^^

TorchOptics provides many profile functions in :mod:`torchoptics.profiles` that generate common
spatial distributions:

.. code-block:: python

    from torchoptics.profiles import gaussian, circle, hermite_gaussian

    # Gaussian beam
    gauss_field = Field(gaussian(200, waist_radius=500e-6))

    # Circular aperture
    circ_field = Field(circle(200, radius=1e-3))

    # Hermite-Gaussian mode (m=1, n=0)
    hg_field = Field(hermite_gaussian(200, m=1, n=0, waist_radius=500e-6))

See :ref:`user-guide-profiles` for the full list of available profiles.

Field Properties
-----------------

Position and Geometry
^^^^^^^^^^^^^^^^^^^^^

Every field has a well-defined geometry described by its shape, spacing, z-position, and offset:

.. code-block:: python

    field = Field(torch.ones(200, 300), z=0.1, spacing=(10e-6, 15e-6), offset=(0, 1e-3))

    print(field.shape)    # (200, 300)
    print(field.z)        # tensor(0.1)
    print(field.spacing)  # tensor([10e-6, 15e-6])
    print(field.offset)   # tensor([0, 1e-3])

The physical extent of the grid is given by:

.. code-block:: python

    print(field.length())   # Physical size: shape × spacing
    print(field.bounds())   # Min and max coordinates along each axis

Intensity and Power
^^^^^^^^^^^^^^^^^^^

The intensity of a field is the squared magnitude :math:`I(x,y) = |U(x,y)|^2`:

.. code-block:: python

    intensity = field.intensity()  # Returns real-valued tensor

The total power is the integrated intensity over the grid area:

.. code-block:: python

    power = field.power()  # Returns scalar tensor

Statistical Properties
^^^^^^^^^^^^^^^^^^^^^^^

You can compute the centroid and standard deviation of the intensity distribution:

.. code-block:: python

    centroid = field.centroid()  # (y_centroid, x_centroid) in physical coordinates
    std = field.std()            # (y_std, x_std) — spatial spread of intensity

These are useful for tracking beam position and size during propagation or optimization.

Field Operations
-----------------

Propagation
^^^^^^^^^^^

TorchOptics provides several methods for free-space propagation:

.. code-block:: python

    # Propagate to a specific z-position, keeping the same grid geometry
    propagated = field.propagate_to_z(z=0.5)

    # Propagate to a specific z with a new grid shape and spacing
    propagated = field.propagate(shape=300, z=0.5, spacing=20e-6)

    # Propagate to match another plane's geometry
    target_plane = PlanarGrid(shape=300, z=0.5, spacing=20e-6)
    propagated = field.propagate_to_plane(target_plane)

See :ref:`user-guide-propagation` for details on propagation methods.

Modulation
^^^^^^^^^^

Modulation multiplies the field by a spatial profile, modeling the effect of optical elements:

.. code-block:: python

    from torchoptics.profiles import circle

    # Apply a circular aperture
    aperture = circle(field.shape, radius=1e-3, spacing=field.spacing)
    apertured_field = field.modulate(aperture)

Normalization
^^^^^^^^^^^^^

Normalize a field to a specified total power:

.. code-block:: python

    # Normalize to unit power
    normalized = field.normalize(normalized_power=1.0)
    print(normalized.power())  # tensor(1.0)

Inner and Outer Products
^^^^^^^^^^^^^^^^^^^^^^^^^

The spatial inner product between two fields computes the overlap integral
:math:`\langle U_1 | U_2 \rangle = \int U_1^*(x,y) \, U_2(x,y) \, dx \, dy`:

.. code-block:: python

    overlap = field1.inner(field2)  # Complex scalar

The outer product :math:`U_1(x_1, y_1) \, U_2^*(x_2, y_2)` is used in coherence calculations:

.. code-block:: python

    coherence = field1.outer(field2)  # 4D tensor

Copying Fields
^^^^^^^^^^^^^^

Create a copy of a field, optionally overriding properties:

.. code-block:: python

    copy = field.copy(z=0.5)  # Same data, different z-position

Batch Processing
-----------------

Fields support batch dimensions, just like PyTorch tensors. Any dimensions before the last two
are treated as batch dimensions:

.. code-block:: python

    # Batch of 10 fields, each 100×100
    batch_data = torch.randn(10, 100, 100, dtype=torch.complex64)
    batch_field = Field(batch_data, wavelength=700e-9, spacing=10e-6)

    # All operations work element-wise across batch dimensions
    propagated_batch = batch_field.propagate_to_z(0.1)
    print(propagated_batch.data.shape)  # torch.Size([10, 100, 100])

This enables efficient parallel simulation of multiple fields simultaneously, which is
particularly useful for parameter sweeps and optimization.

Polarized Fields
^^^^^^^^^^^^^^^^

Polarized fields are represented by making the third-to-last dimension have size 3, corresponding
to the :math:`(E_x, E_y, E_z)` components of the electric field:

.. code-block:: python

    # x-polarized field
    polarized_data = torch.zeros(3, 100, 100)
    polarized_data[0] = 1  # x-component only
    field = Field(polarized_data)

See :ref:`user-guide-polarization` for more on polarized light simulation.

Visualization
--------------

The :meth:`~torchoptics.Field.visualize` method provides a convenient way to visualize fields:

.. code-block:: python

    # Basic visualization
    field.visualize(title="My Field")

    # Show intensity only
    field.visualize(intensity=True, title="Intensity")

For complex fields, ``visualize()`` shows two subplots: the squared magnitude (intensity) and the
phase. For real fields, it shows a single plot.

When working with batch or polarized fields, use index arguments to select a specific slice:

.. code-block:: python

    # Visualize the 3rd field in a batch
    batch_field.visualize(2, title="Third field in batch")

    # Visualize the x-component of a polarized field
    polarized_field.visualize(0, title="Ex component")

See also :func:`~torchoptics.visualize_tensor` and :func:`~torchoptics.animate_tensor` for
lower-level visualization utilities.
