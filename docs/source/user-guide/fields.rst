Fields
======


The :class:`~torchoptics.Field` class is the central object in TorchOptics: a complex-valued
wavefront sampled on a 2D planar grid at a position along the optical axis.


Creating a Field
----------------

Construct a :class:`~torchoptics.Field` from a 2D complex tensor. If ``spacing`` or
``wavelength`` are omitted, the global defaults are used (see :doc:`configuration`).

.. plot::
    :context: reset

    import torch
    import torchoptics
    from torchoptics import Field
    from torchoptics.profiles import gaussian, circle

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    field = Field(circle(300, radius=1e-3))
    field.visualize(title="Circular Aperture")
    print(field)

The ``data`` tensor must have at least 2 dimensions (H × W). Leading dimensions are treated as
batch dimensions. Fields can be created from :mod:`~torchoptics.profiles` functions (see
:doc:`profiles`) or from arbitrary tensors:

.. plot::
    :context: close-figs

    gaussian_field = Field(gaussian(300, waist_radius=500e-6))
    gaussian_field.visualize(title="Gaussian Beam")


Grid Geometry
-------------

Every :class:`~torchoptics.Field` inherits from :class:`~torchoptics.PlanarGrid`, which defines
its spatial layout through four properties:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Property
     - Type
     - Description
   * - ``shape``
     - ``(int, int)``
     - Number of grid points (H, W).
   * - ``spacing``
     - ``Tensor``
     - Physical distance between adjacent grid points (m). Can differ along the two axes.
   * - ``offset``
     - ``Tensor``
     - The (y, x) coordinates of the grid center (m). Default: ``(0, 0)``.
   * - ``z``
     - ``Tensor``
     - Position along the optical axis (m). Default: ``0``.

Retrieve spatial coordinates and bounds:

.. code-block:: python

    x, y = field.meshgrid()      # 2D coordinate arrays
    bounds = field.bounds()       # [y_min, y_max, x_min, x_max]
    length = field.length()       # Physical extent [Ly, Lx]


Propagation
-----------

Three methods handle free-space propagation (see :doc:`propagation` for the underlying
algorithms):

:meth:`~torchoptics.Field.propagate_to_z` — propagate to a new ``z`` while preserving grid
geometry:

.. plot::
    :context: close-figs

    propagated = field.propagate_to_z(0.5)
    propagated.visualize(title="Propagated to z = 0.5 m")

:meth:`~torchoptics.Field.propagate` — full control over the output grid:

.. code-block:: python

    output = field.propagate(shape=(512, 512), z=1.0, spacing=5e-6, offset=(100e-6, 0))

:meth:`~torchoptics.Field.propagate_to_plane` — propagate to a
:class:`~torchoptics.PlanarGrid` or element:

.. code-block:: python

    from torchoptics import PlanarGrid

    target = PlanarGrid(shape=400, z=0.3, spacing=8e-6)
    output = field.propagate_to_plane(target)

All three accept optional ``propagation_method``, ``asm_pad``, and ``interpolation_mode``
keyword arguments.


Analysis
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - :meth:`~torchoptics.Field.intensity`
     - Squared magnitude :math:`|\psi|^2`.
   * - :meth:`~torchoptics.Field.power`
     - Integrated intensity: :math:`P = \sum I_{ij}\,\Delta A`.
   * - :meth:`~torchoptics.Field.centroid`
     - Intensity-weighted center of mass :math:`(\bar{y}, \bar{x})`.
   * - :meth:`~torchoptics.Field.std`
     - Intensity-weighted standard deviation along each axis.

.. plot::
    :context: close-figs

    g = Field(gaussian(300, waist_radius=500e-6))
    print(f"Power:    {g.power().item():.4e}")
    print(f"Centroid: ({g.centroid()[0].item():.2e}, {g.centroid()[1].item():.2e})")
    print(f"Std:      ({g.std()[0].item():.2e}, {g.std()[1].item():.2e})")


Inner Product
-------------

The overlap integral between two fields is:

.. math::

    \langle \psi_1 | \psi_2 \rangle = \sum_{i,j} \psi_1^*(i,j) \, \psi_2(i,j) \, \Delta A

:meth:`~torchoptics.Field.inner` returns this as a complex scalar. Taking the squared magnitude
gives the **mode overlap**: a value in :math:`[0, 1]` when both fields are normalized to unit
power, and a natural loss function for inverse design (see :doc:`inverse_design`):

.. code-block:: python

    overlap = field_a.inner(field_b).abs().square()  # in [0, 1]
    loss = 1 - overlap

Both fields must share the same geometry.

Modulation
----------

Point-wise complex multiplication:

.. math::

    \psi'(x, y) = \mathcal{M}(x, y) \cdot \psi(x, y)

This is the mechanism by which :doc:`elements <elements>` transform fields, and can be used to apply 
arbitrary complex-valued masks:

.. code-block:: python

    profile = torch.exp(1j * torch.randn(300, 300, dtype=torch.double))
    modulated = field.modulate(profile)


Normalization and Copying
-------------------------

.. code-block:: python

    normalized = field.normalize()           # Scale to unit power
    scaled = field.normalize(2.5)            # Scale to power = 2.5
    shifted = field.copy(z=0.5)              # Copy with updated z
    rescaled = field.copy(spacing=5e-6)      # Copy with new spacing


Updating Properties
-------------------

All registered properties can be updated by direct assignment.
Assignments are validated automatically; invalid values raise errors immediately.

.. code-block:: python

    field.z = 0.5
    field.wavelength = 532e-9
    field.spacing = (8e-6, 8e-6)
    field.offset = (100e-6, 0)

Element properties work the same way; see :ref:`custom-elements`.


Visualization
-------------

:meth:`~torchoptics.Field.visualize` displays intensity and phase for complex fields:

.. plot::
    :context: close-figs

    from torchoptics.profiles import laguerre_gaussian

    lg = Field(laguerre_gaussian(300, p=1, l=2, waist_radius=500e-6))
    lg.visualize(title="LG$_{1}^{2}$ Mode")

The standalone :func:`~torchoptics.visualize_tensor` and :func:`~torchoptics.animate_tensor`
functions work with arbitrary 2D and 3D tensors respectively.


Batched Fields
--------------

The ``data`` tensor supports batch dimensions with shape ``(..., H, W)``:

.. code-block:: python

    batch_data = torch.randn(4, 300, 300, dtype=torch.cdouble)
    batch_field = Field(batch_data)

    propagated = batch_field.propagate_to_z(0.5)
    print(propagated.data.shape)  # torch.Size([4, 300, 300])

