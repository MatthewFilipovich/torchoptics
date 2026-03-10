.. _user-guide-propagation:

Propagation Methods
====================

TorchOptics simulates the propagation of optical fields through free space using numerical methods
based on scalar diffraction theory. The library provides two primary propagation methods and an
automatic selection mode.

.. contents:: On This Page
   :local:
   :depth: 2

Physics Background
-------------------

When a monochromatic optical field :math:`U(x, y, 0)` propagates through free space to a plane at
distance :math:`z`, the resulting field :math:`U(x, y, z)` is described by the Rayleigh-Sommerfeld
diffraction integral:

.. math::
    U(x, y, z) = \frac{z}{i\lambda} \iint U(x', y', 0) \, \frac{\exp(ikr)}{r^2} \, dx' \, dy'

where :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + z^2}` and :math:`k = 2\pi/\lambda`.

TorchOptics implements two numerical approaches to evaluate this integral, along with an automatic
selection mode that chooses the most efficient one.


Angular Spectrum Method (ASM)
------------------------------

The Angular Spectrum Method computes propagation in the Fourier domain using a transfer function.
The field is decomposed into plane-wave components, each propagated independently, then recombined:

.. math::
    U(x, y, z) = \mathcal{F}^{-1}\!\left\{\mathcal{F}\{U(x, y, 0)\} \cdot H(f_x, f_y, z)\right\}

where the transfer function is:

.. math::
    H(f_x, f_y, z) = \exp\!\left(i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2}\right)

**Advantages:**

- Very efficient — uses FFT operations (:math:`O(N \log N)`)
- Accurate for large propagation distances
- Supports zero-padding to reduce aliasing artifacts

**Limitations:**

- Input and output grids must have the same spacing
- May exhibit aliasing for very short propagation distances

Use ``propagation_method="ASM"`` to force the angular spectrum method:

.. code-block:: python

    output = field.propagate_to_z(z=0.5, propagation_method="ASM")

ASM Padding
^^^^^^^^^^^^

By default, the ASM pads the field to reduce edge artifacts. The ``asm_pad`` parameter controls
the amount of padding relative to the field size:

.. code-block:: python

    # Double padding (default)
    output = field.propagate_to_z(z=0.5, propagation_method="ASM", asm_pad=2)

    # No padding (faster but more edge artifacts)
    output = field.propagate_to_z(z=0.5, propagation_method="ASM", asm_pad=0)

.. tip::
    For 1D simulations (one spatial dimension set to 1), set ``asm_pad=0`` for the degenerate
    dimension.


Direct Integration Method (DIM)
---------------------------------

The Direct Integration Method directly evaluates the Rayleigh-Sommerfeld integral by computing
the impulse response between every pair of input and output grid points:

.. math::
    U(x, y, z) = \sum_{x', y'} U(x', y', 0) \, h(x - x', y - y', z) \, \Delta x' \, \Delta y'

where :math:`h` is the free-space impulse response.

**Advantages:**

- Supports different input and output grid geometries (shape, spacing, offset)
- More accurate for short propagation distances
- No aliasing artifacts

**Limitations:**

- Computationally expensive — :math:`O(N_{\text{in}} \times N_{\text{out}})` complexity
- Slower than ASM for large grids

Use ``propagation_method="DIM"`` to force direct integration:

.. code-block:: python

    output = field.propagate(shape=300, z=0.5, spacing=20e-6, propagation_method="DIM")


Rayleigh-Sommerfeld Variants
------------------------------

Both ASM and DIM have Rayleigh-Sommerfeld variants (``ASM_RS`` and ``DIM_RS``) that use a
slightly different formulation of the diffraction integral. These variants include an additional
obliquity factor and may provide improved accuracy for certain geometries:

.. code-block:: python

    output = field.propagate_to_z(z=0.5, propagation_method="ASM_RS")
    output = field.propagate(shape=300, z=0.5, propagation_method="DIM_RS")

.. important::
    RS propagation methods are more sensitive to floating-point precision. If you encounter
    numerical artifacts when using ``ASM_RS``, ``DIM_RS``, or ``AUTO_RS``, switch to double
    precision:

    .. code-block:: python

        import torch
        torch.set_default_dtype(torch.float64)

    See :ref:`user-guide-precision` for details.


Automatic Method Selection
----------------------------

The default ``propagation_method="AUTO"`` automatically selects between ASM and DIM based on
the **critical propagation distance**. This is the distance at which ASM becomes accurate relative
to DIM for the given grid parameters:

.. code-block:: python

    from torchoptics.propagation import calculate_critical_propagation_distance

    z_critical = calculate_critical_propagation_distance(field, output_plane)
    print(f"Critical distance: {z_critical:.4f} m")

When the propagation distance exceeds the critical distance, ASM is used; otherwise, DIM is selected.

Similarly, ``AUTO_RS`` selects between ``ASM_RS`` and ``DIM_RS``:

.. code-block:: python

    output = field.propagate_to_z(z=0.5, propagation_method="AUTO_RS")


Propagation API
----------------

propagate_to_z
^^^^^^^^^^^^^^^

The simplest propagation method — propagates to a target :math:`z`-position while keeping the
same grid shape, spacing, and offset:

.. code-block:: python

    output = field.propagate_to_z(z=0.5)

propagate
^^^^^^^^^^^

Full control over the output geometry — specify a new shape, z, spacing, and offset:

.. code-block:: python

    output = field.propagate(
        shape=300,
        z=0.5,
        spacing=20e-6,
        offset=(0, 1e-3),
        propagation_method="AUTO",
    )

propagate_to_plane
^^^^^^^^^^^^^^^^^^^

Propagate to match the geometry of an existing :class:`~torchoptics.PlanarGrid`:

.. code-block:: python

    from torchoptics import PlanarGrid

    target = PlanarGrid(shape=300, z=0.5, spacing=20e-6)
    output = field.propagate_to_plane(target)

Interpolation Modes
^^^^^^^^^^^^^^^^^^^^

When the output grid doesn't match the input grid spacing, TorchOptics can interpolate the
result. The ``interpolation_mode`` parameter controls this:

- ``"none"`` — No interpolation (default for matching grids)
- ``"nearest"`` — Nearest-neighbor interpolation (default)
- ``"bilinear"`` — Bilinear interpolation
- ``"bicubic"`` — Bicubic interpolation

.. code-block:: python

    output = field.propagate(
        shape=300, z=0.5, spacing=20e-6,
        interpolation_mode="bilinear",
    )


Choosing the Right Method
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Scenario
     - Recommended Method
     - Reason
   * - Large propagation distance
     - ``ASM`` or ``AUTO``
     - Fast FFT-based computation, no aliasing at large z
   * - Short propagation distance
     - ``DIM`` or ``AUTO``
     - Avoids ASM aliasing at small z
   * - Different input/output grids
     - ``DIM``
     - ASM requires same spacing on both grids
   * - Maximum performance
     - ``ASM`` with small ``asm_pad``
     - Minimizes FFT size
   * - Unsure which to use
     - ``AUTO`` (default)
     - Automatically selects the best method


.. _user-guide-precision:

Floating-Point Precision
-------------------------

By default, TorchOptics uses single precision (``float32``), matching PyTorch's default. This is
sufficient for most simulations and provides better performance, especially on GPUs.

However, some simulations — particularly those using Rayleigh-Sommerfeld propagation methods
(``ASM_RS``, ``DIM_RS``, ``AUTO_RS``) — may require double precision (``float64``) for accurate
results. The RS formulation involves computing :math:`\sqrt{x^2 + y^2 + z^2}`, which can
lose significant digits in single precision when the terms have very different magnitudes.

To enable double precision, set the default dtype at the start of your script:

.. code-block:: python

    import torch
    torch.set_default_dtype(torch.float64)

.. tip::
    If you observe unexpected artifacts or inaccurate results, switching to ``float64`` is a
    good first debugging step.
