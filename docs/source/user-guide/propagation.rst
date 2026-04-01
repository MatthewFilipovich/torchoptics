Propagation
===========


TorchOptics simulates free-space propagation using Fourier optics. Two numerical methods are
available: the **angular spectrum method** (ASM) and the **direct integration method** (DIM).
Each can use either the exact **Rayleigh–Sommerfeld** diffraction model or the **Fresnel
approximation**.


Automatic Method Selection
---------------------------

By default (``propagation_method="AUTO"``), TorchOptics selects the optimal method by computing
a critical distance :math:`z_c` and comparing it to the propagation distance
:math:`|\Delta z|`:

.. math::

    z_c = \frac{2 \, |x_\text{max}| \, \Delta}{\lambda}

where :math:`|x_\text{max}|` denotes, for each transverse dimension (x and y), the maximum
coordinate separation along that dimension between any grid point in the input plane and
any grid point in the propagation plane; :math:`\Delta` is the grid spacing in that
dimension, and :math:`\lambda` is the wavelength.

- :math:`|\Delta z| < z_c` in at least one dimension → **ASM** (short-distance regime).
- Otherwise → **DIM** (long-distance regime).

Override the selection explicitly:

.. code-block:: python

    field.propagate_to_z(0.1, propagation_method="ASM")
    field.propagate_to_z(2.0, propagation_method="DIM")


Angular Spectrum Method (ASM)
-----------------------------

ASM propagates in the frequency domain by multiplying the field's Fourier transform by a
transfer function:

.. math::

    \psi_\text{out}(x,y) = \mathcal{F}^{-1}\!\bigl\{
    H(k_x, k_y) \cdot \mathcal{F}\{\psi_\text{in}\}\bigr\}

**Rayleigh–Sommerfeld** transfer function (default):

.. math::

    H(k_x, k_y) = \exp\!\left(i \sqrt{k^2 - k_x^2 - k_y^2} \cdot \Delta z\right)

**Fresnel approximation** (``"ASM_FRESNEL"``):

.. math::

    H(k_x, k_y) = \exp(ik\Delta z) \cdot
    \exp\!\left(-i \frac{\lambda \Delta z}{4\pi}(k_x^2 + k_y^2)\right)

**Padding.** ASM zero-pads the field before the FFT to suppress periodic boundary artifacts.
The default pad is twice the field size; override with ``asm_pad``:

.. code-block:: python

    field.propagate_to_z(0.1, asm_pad=(300, 300))   # custom padding
    field.propagate_to_z(0.1, asm_pad=(0, 0))       # no padding (faster)


Direct Integration Method (DIM)
--------------------------------

DIM computes propagation as a convolution with the free-space impulse response:

.. math::

    \psi_\text{out}(x,y) = \iint h(x-x', y-y') \, \psi_\text{in}(x', y') \, dx'\,dy'

**Rayleigh–Sommerfeld** impulse response (default):

.. math::

    h(x,y) = \frac{1}{2\pi}\!\left(\frac{1}{r} - ik\right)
    \frac{\Delta z}{r^2} \, e^{ikr} \, \Delta A
    \qquad\text{with } r = \sqrt{x^2 + y^2 + \Delta z^2}

**Fresnel approximation** (``"DIM_FRESNEL"``):

.. math::

    h(x,y) = \frac{e^{ik\Delta z}}{i\lambda\Delta z}
    \exp\!\left(\frac{ik}{2\Delta z}(x^2 + y^2)\right) \Delta A

The convolution is computed via :func:`~torchoptics.functional.conv2d_fft`.


Method Summary
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - String
     - Algorithm
     - Physical Model
   * - ``"AUTO"``
     - Auto-select ASM or DIM
     - Rayleigh–Sommerfeld
   * - ``"AUTO_FRESNEL"``
     - Auto-select ASM or DIM
     - Fresnel approximation
   * - ``"ASM"``
     - Angular spectrum method
     - Rayleigh–Sommerfeld
   * - ``"ASM_FRESNEL"``
     - Angular spectrum method
     - Fresnel approximation
   * - ``"DIM"``
     - Direct integration method
     - Rayleigh–Sommerfeld
   * - ``"DIM_FRESNEL"``
     - Direct integration method
     - Fresnel approximation


When to Use Which
-----------------

- **ASM** — best for short-distance propagation where the output grid has similar spatial
  extent to the input. Fast (FFT-based) but requires padding.

- **DIM** — best for long-distance propagation or when the output grid has very different
  spacing or offset. Can change grid geometry without interpolation artifacts.

- **Fresnel variants** (``"ASM_FRESNEL"``, ``"DIM_FRESNEL"``, ``"AUTO_FRESNEL"``) — use when
  the paraxial approximation holds (:math:`\theta \ll 1` rad) and you need faster computation
  or match a specific physical model. For high-NA systems or large propagation angles, the
  default Rayleigh–Sommerfeld is more accurate.

For most use cases, ``"AUTO"`` works well.


Interpolation
-------------

When the output plane has different ``spacing`` or ``offset`` from the propagation plane,
TorchOptics resamples the result. Control this with ``interpolation_mode``:

.. code-block:: python

    output = field.propagate(
        shape=512, z=0.5, spacing=5e-6,
        interpolation_mode="bilinear",
    )

Available modes: ``"nearest"`` (default), ``"bilinear"``, ``"bicubic"``, ``"none"``.


Debug Logging
-------------

Enable debug logging to inspect method selection and plane geometries:

.. code-block:: python

    import logging
    logging.getLogger("torchoptics").setLevel(logging.DEBUG)

    field.propagate_to_z(0.5)
