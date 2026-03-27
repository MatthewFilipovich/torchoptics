Profiles
========


The :mod:`torchoptics.profiles` module provides functions for generating common spatial profiles.
Each function returns a 2D tensor that can be used as the ``data`` argument to
:class:`~torchoptics.Field` or as input to elements like
:class:`~torchoptics.elements.AmplitudeModulator`.

All profile functions accept ``shape``, ``spacing``, and ``offset`` arguments. When ``spacing``
is omitted, the global default is used.


Beam Modes
----------

Gaussian
^^^^^^^^

:func:`~torchoptics.profiles.gaussian` generates the fundamental Gaussian beam:

.. plot::
    :context: reset
    :nofigs:

    import torch
    import torchoptics
    from torchoptics import visualize_tensor
    from torchoptics.profiles import (
    airy, airy_beam, bessel, binary_grating, blazed_grating, checkerboard,
    circle, cylindrical_lens_phase, gaussian, hermite_gaussian, hexagon,
    laguerre_gaussian, lens_phase, plane_wave_phase, rectangle,
    siemens_star, sinc, sinusoidal_grating, spherical_wave_phase,
    square, triangle, zernike,
    )

    torchoptics.set_default_spacing(10e-6)

.. plot::
    :context: close-figs

    profile = gaussian(300, waist_radius=500e-6)
    visualize_tensor(profile, title="Gaussian Beam")

Hermite-Gaussian
^^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.hermite_gaussian` generates higher-order :math:`\text{HG}_{mn}`
modes:

.. plot::
    :context: close-figs

    profile = hermite_gaussian(300, m=2, n=1, waist_radius=500e-6)
    visualize_tensor(profile, title="HG$_{2,1}$ Mode")

Laguerre-Gaussian
^^^^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.laguerre_gaussian` generates :math:`\text{LG}_{p}^{l}` modes,
which carry orbital angular momentum when :math:`l \neq 0`:

.. plot::
    :context: close-figs

    profile = laguerre_gaussian(300, p=0, l=3, waist_radius=500e-6)
    visualize_tensor(profile, title="LG$_{0}^{3}$ Mode (Vortex Beam)")

Superpositions create structured beams:

.. plot::
    :context: close-figs

    petal = laguerre_gaussian(300, p=0, l=4, waist_radius=500e-6) \
        + laguerre_gaussian(300, p=0, l=-4, waist_radius=500e-6)
    visualize_tensor(petal, title="8-Petal Beam")

Bessel
^^^^^^

:func:`~torchoptics.profiles.bessel` generates non-diffracting Bessel beams
:math:`J_0(k r \sin\theta)`:

.. plot::
    :context: close-figs

    profile = bessel(300, cone_angle=0.01, wavelength=700e-9)
    vmax = profile.abs().max() * 0.5
    visualize_tensor(profile, title="Bessel Beam", vmin=-vmax, vmax=vmax, cmap="RdBu")

Airy Beam
^^^^^^^^^

:func:`~torchoptics.profiles.airy_beam` generates a truncated 2D Airy beam,
which combines the Airy function with an exponential truncation factor to keep
the energy finite while preserving the characteristic self-accelerating lobe:

.. plot::
    :context: close-figs

    profile = airy_beam(300, scale=50e-6, truncation=0.05)
    vmax = -profile.abs().max().item() * 0.5
    visualize_tensor(profile, title="Airy Beam", vmin=-vmax, vmax=vmax, cmap="RdBu")


Geometric Apertures
-------------------

Binary aperture masks (1 inside, 0 outside):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`~torchoptics.profiles.circle`
     - Circular aperture with given ``radius``.
   * - :func:`~torchoptics.profiles.rectangle`
     - Rectangular aperture with side lengths ``side``.
   * - :func:`~torchoptics.profiles.square`
     - Square aperture with side length ``side``.
   * - :func:`~torchoptics.profiles.triangle`
     - Triangular aperture with given ``base`` and ``height``.
   * - :func:`~torchoptics.profiles.hexagon`
     - Regular hexagon with circumradius ``radius``.
   * - :func:`~torchoptics.profiles.octagon`
     - Regular octagon with circumradius ``radius``.
   * - :func:`~torchoptics.profiles.regular_polygon`
     - Regular N-sided polygon with circumradius ``radius``.

.. plot::
    :context: close-figs

    profile = circle(300, radius=1e-3)
    visualize_tensor(profile, title="Circle")

.. plot::
    :context: close-figs

    profile = rectangle(300, side=(1.5e-3, 0.8e-3))
    visualize_tensor(profile, title="Rectangle")

.. plot::
    :context: close-figs

    profile = square(300, side=1.5e-3)
    visualize_tensor(profile, title="Square")

.. plot::
    :context: close-figs

    profile = triangle(300, base=1.5e-3, height=1.5e-3)
    visualize_tensor(profile, title="Triangle")

.. plot::
    :context: close-figs

    profile = hexagon(300, radius=1e-3)
    visualize_tensor(profile, title="Hexagon")


Gratings
--------

Periodic profiles along a configurable direction:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - :func:`~torchoptics.profiles.binary_grating`
     - Square-wave grating with configurable duty cycle.
   * - :func:`~torchoptics.profiles.blazed_grating`
     - Sawtooth (linearly ramped) grating.
   * - :func:`~torchoptics.profiles.sinusoidal_grating`
     - Smooth sinusoidal grating.

.. plot::
    :context: close-figs

    profile = binary_grating(300, period=200e-6)
    visualize_tensor(profile, title="Binary Grating")

.. plot::
    :context: close-figs

    profile = blazed_grating(300, period=200e-6)
    visualize_tensor(profile, title="Blazed Grating")

.. plot::
    :context: close-figs

    profile = sinusoidal_grating(300, period=200e-6)
    visualize_tensor(profile, title="Sinusoidal Grating")


Test Patterns
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`~torchoptics.profiles.checkerboard`
     - Tiled checkerboard pattern.
   * - :func:`~torchoptics.profiles.siemens_star`
     - Spoke-based resolution target.

.. plot::
    :context: close-figs

    profile = checkerboard(300, tile_length=200e-6, num_tiles=10)
    visualize_tensor(profile, title="Checkerboard")

.. plot::
    :context: close-figs

    profile = siemens_star(300, num_spokes=24, radius=1.2e-3)
    visualize_tensor(profile, title="Siemens Star")


Wave Phases
-----------

Real-valued phase tensors. Apply as ``torch.exp(1j * phase)`` to create a complex field:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - :func:`~torchoptics.profiles.plane_wave_phase`
     - Tilted plane wave with polar angle :math:`\theta` and azimuthal angle :math:`\phi`.
   * - :func:`~torchoptics.profiles.spherical_wave_phase`
     - Diverging spherical wave from a point source.
   * - :func:`~torchoptics.profiles.lens_phase`
     - Quadratic thin-lens phase profile.
   * - :func:`~torchoptics.profiles.cylindrical_lens_phase`
     - Quadratic phase in one direction.

.. plot::
    :context: close-figs

    field = torch.exp(1j * plane_wave_phase(300, theta=0.001, wavelength=700e-9))
    visualize_tensor(field, title="Plane Wave Phase")

.. plot::
    :context: close-figs

    field = torch.exp(1j * spherical_wave_phase(300, z=0.5, wavelength=700e-9))
    visualize_tensor(field, title="Spherical Wave Phase")

.. plot::
    :context: close-figs

    field = torch.exp(1j * lens_phase(300, focal_length=300e-3, wavelength=700e-9))
    visualize_tensor(field, title="Lens Phase")

.. plot::
    :context: close-figs

    field = torch.exp(1j * cylindrical_lens_phase(300, focal_length=300e-3, wavelength=700e-9))
    visualize_tensor(field, title="Cylindrical Lens Phase")


Special Functions
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`~torchoptics.profiles.airy`
     - Airy pattern :math:`\bigl(2J_1(r/a)/(r/a)\bigr)^2`.
   * - :func:`~torchoptics.profiles.sinc`
     - 2D sinc function (Fourier transform of a rectangle).
   * - :func:`~torchoptics.profiles.zernike`
     - `Zernike polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_
       :math:`Z_n^m(\rho, \theta)` for wavefront aberrations.

.. plot::
    :context: close-figs

    profile = airy(300, scale=100e-6)
    visualize_tensor(profile, title="Airy Pattern")

.. plot::
    :context: close-figs

    profile = sinc(300, scale=(500e-6, 500e-6))
    visualize_tensor(profile, title="Sinc")

.. plot::
    :context: close-figs

    data = zernike(300, n=2, m=2, radius=1.4e-3)
    visualize_tensor(
        data,
        title="Zernike $Z_2^2$ (Astigmatism)",
        cmap="RdBu",
        vmin=-data.max().item(),
        vmax=+data.max().item(),
    )


Coherence Functions
-------------------

4D tensors representing the mutual coherence function
:math:`\Gamma(x_1, y_1, x_2, y_2)` for partially coherent light (see
:doc:`spatial_coherence`):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - :func:`~torchoptics.profiles.schell_model`
     - General Schell model with custom intensity and coherence functions.
   * - :func:`~torchoptics.profiles.gaussian_schell_model`
     - Gaussian Schell model with Gaussian intensity and coherence.


Using Profiles with Elements
-----------------------------

Profiles plug directly into elements:

.. code-block:: python

    import math
    from torchoptics.elements import AmplitudeModulator, PhaseModulator
    from torchoptics.profiles import circle, blazed_grating

    aperture = AmplitudeModulator(circle(300, radius=1e-3), z=0.1)
    grating = PhaseModulator(blazed_grating(300, period=100e-6, height=2 * math.pi), z=0.2)
