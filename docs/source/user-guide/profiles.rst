.. _user-guide-profiles:

Spatial Profiles
=================

The :mod:`torchoptics.profiles` module provides functions for generating common spatial
distributions used to initialize optical fields, define modulation patterns, and create
test targets. All profile functions return :class:`torch.Tensor` objects.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

Profile functions generate 2D tensor distributions on a planar grid. They all share a common
interface pattern:

.. code-block:: python

    profile = profile_function(shape, ..., spacing=None, offset=None)

- **shape**: Grid size — an integer ``N`` for an ``N×N`` grid or a tuple ``(H, W)``
- **spacing**: Grid spacing. Uses the global default if not specified.
- **offset**: Center coordinates of the grid. Defaults to ``(0, 0)``.

The returned tensor can be passed directly to :class:`~torchoptics.Field`:

.. code-block:: python

    from torchoptics import Field
    from torchoptics.profiles import gaussian

    field = Field(gaussian(200, waist_radius=500e-6))


Geometric Shapes
-----------------

Binary shapes that are 1 inside the shape boundary and 0 outside. Useful for apertures, masks,
and test targets.

Circle
^^^^^^^

:func:`~torchoptics.profiles.circle` — Binary circular aperture:

.. code-block:: python

    from torchoptics.profiles import circle

    # Circle with 1 mm radius on a 200×200 grid
    circ = circle(200, radius=1e-3)

Rectangle
^^^^^^^^^^

:func:`~torchoptics.profiles.rectangle` — Binary rectangular aperture:

.. code-block:: python

    from torchoptics.profiles import rectangle

    # 2 mm × 1 mm rectangle
    rect = rectangle(200, side=(2e-3, 1e-3))

Square
^^^^^^^

:func:`~torchoptics.profiles.square` — Binary square aperture:

.. code-block:: python

    from torchoptics.profiles import square

    sq = square(200, side=1.5e-3)

Triangle
^^^^^^^^^

:func:`~torchoptics.profiles.triangle` — Binary triangular aperture with optional rotation:

.. code-block:: python

    from torchoptics.profiles import triangle

    # Triangle with 2 mm base, 1 mm height
    tri = triangle(200, base=2e-3, height=1e-3)

    # Rotated triangle (45 degrees)
    tri_rot = triangle(200, base=2e-3, height=1e-3, theta=torch.pi / 4)

Checkerboard
^^^^^^^^^^^^^

:func:`~torchoptics.profiles.checkerboard` — Alternating binary checkerboard pattern:

.. code-block:: python

    from torchoptics.profiles import checkerboard

    # 15×15 tiles, each 400 μm wide
    cb = checkerboard(1000, tile_length=400e-6, num_tiles=15)


Beam Profiles
--------------

Gaussian Beam
^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.gaussian` — Fundamental Gaussian beam (TEM₀₀ mode):

.. math::
    U(x, y) = \frac{w_0}{w(z)} \exp\!\left(-\frac{x^2 + y^2}{w^2(z)}\right)
    \exp\!\left(-\frac{ik(x^2 + y^2)}{2R(z)} + i\zeta(z)\right)

where :math:`w(z)` is the beam width, :math:`R(z)` is the radius of curvature, and
:math:`\zeta(z)` is the Gouy phase.

.. code-block:: python

    from torchoptics.profiles import gaussian

    # Gaussian beam at its waist
    g = gaussian(200, waist_radius=500e-6)

    # Gaussian at waist_z=0, evaluated at z=0.1 m (diverging beam)
    g_div = gaussian(200, waist_radius=500e-6, waist_z=0.1)

Hermite-Gaussian Modes
^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.hermite_gaussian` — Higher-order Hermite-Gaussian modes :math:`\text{HG}_{mn}`:

.. code-block:: python

    from torchoptics.profiles import hermite_gaussian

    # Fundamental mode (same as gaussian)
    hg00 = hermite_gaussian(200, m=0, n=0, waist_radius=500e-6)

    # HG₁₀ mode
    hg10 = hermite_gaussian(200, m=1, n=0, waist_radius=500e-6)

    # HG₂₁ mode
    hg21 = hermite_gaussian(200, m=2, n=1, waist_radius=500e-6)

    # Beam at a specific z-position (includes Gouy phase and divergence)
    hg10_prop = hermite_gaussian(200, m=1, n=0, waist_radius=500e-6, waist_z=0.1)

Laguerre-Gaussian Modes
^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.laguerre_gaussian` — Laguerre-Gaussian modes :math:`\text{LG}_{pl}` with
orbital angular momentum:

.. code-block:: python

    from torchoptics.profiles import laguerre_gaussian

    # LG₀₀ (same as Gaussian)
    lg00 = laguerre_gaussian(200, p=0, l=0, waist_radius=500e-6)

    # LG₀₁ — carries orbital angular momentum ℓ=1
    lg01 = laguerre_gaussian(200, p=0, l=1, waist_radius=500e-6)

    # LG₁₂ — radial order p=1, azimuthal order l=2
    lg12 = laguerre_gaussian(200, p=1, l=2, waist_radius=500e-6)

Bessel Beam
^^^^^^^^^^^^

:func:`~torchoptics.profiles.bessel` — Zeroth-order Bessel beam, characterized by its cone angle:

.. code-block:: python

    from torchoptics.profiles import bessel

    # Bessel beam with 0.01 radian cone angle
    b = bessel(200, cone_angle=0.01)

Bessel beams are non-diffracting — their transverse intensity profile is invariant under propagation.


Wave Phases
------------

Plane Wave Phase
^^^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.plane_wave_phase` — Phase of a tilted plane wave, parameterized by
polar angle :math:`\theta` and azimuthal angle :math:`\phi`:

.. code-block:: python

    from torchoptics.profiles import plane_wave_phase

    # On-axis plane wave (zero phase)
    pw = plane_wave_phase(200, theta=0, phi=0)

    # Tilted plane wave
    pw_tilt = plane_wave_phase(200, theta=0.01, phi=torch.pi / 4)

Spherical Wave Phase
^^^^^^^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.spherical_wave_phase` — Phase of a diverging spherical wave:

.. code-block:: python

    from torchoptics.profiles import spherical_wave_phase

    # Spherical wave diverging from z=0.1 m
    sw = spherical_wave_phase(200, z=0.1)


Lens Phases
------------

Lens Phase
^^^^^^^^^^^

:func:`~torchoptics.profiles.lens_phase` — Quadratic phase imparted by a thin lens:

.. math::
    \varphi(x, y) = -\frac{\pi(x^2 + y^2)}{\lambda f}

.. code-block:: python

    from torchoptics.profiles import lens_phase

    lp = lens_phase(200, focal_length=0.2)

Cylindrical Lens Phase
^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.cylindrical_lens_phase` — Phase of a cylindrical lens oriented at angle
:math:`\theta`:

.. code-block:: python

    from torchoptics.profiles import cylindrical_lens_phase

    clp = cylindrical_lens_phase(200, focal_length=0.2, theta=0)


Diffraction Gratings
---------------------

Binary Grating
^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.binary_grating` — Periodic binary grating with controllable period,
height, orientation, and duty cycle:

.. code-block:: python

    from torchoptics.profiles import binary_grating

    bg = binary_grating(200, period=100e-6)
    bg_custom = binary_grating(200, period=100e-6, height=0.5, duty_cycle=0.3, theta=torch.pi / 6)

Blazed Grating
^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.blazed_grating` — Sawtooth (blazed) grating profile:

.. code-block:: python

    from torchoptics.profiles import blazed_grating

    blz = blazed_grating(200, period=100e-6)

Sinusoidal Grating
^^^^^^^^^^^^^^^^^^^^

:func:`~torchoptics.profiles.sinusoidal_grating` — Smooth sinusoidal grating:

.. code-block:: python

    from torchoptics.profiles import sinusoidal_grating

    sg = sinusoidal_grating(200, period=100e-6)


Zernike Polynomials
--------------------

:func:`~torchoptics.profiles.zernike` — Zernike polynomials :math:`Z_n^m`, commonly used to
describe wavefront aberrations:

.. code-block:: python

    from torchoptics.profiles import zernike

    # Defocus (n=2, m=0)
    defocus = zernike(200, n=2, m=0, radius=1e-3)

    # Vertical astigmatism (n=2, m=2)
    astigmatism = zernike(200, n=2, m=2, radius=1e-3)

    # Vertical coma (n=3, m=1)
    coma = zernike(200, n=3, m=1, radius=1e-3)

    # Spherical aberration (n=4, m=0)
    spherical = zernike(200, n=4, m=0, radius=1e-3)

The order indices must satisfy :math:`|m| \leq n` and :math:`n - |m|` must be even.


Special Functions
------------------

Airy Pattern
^^^^^^^^^^^^^

:func:`~torchoptics.profiles.airy` — Airy diffraction pattern (far-field diffraction of a
circular aperture):

.. code-block:: python

    from torchoptics.profiles import airy

    a = airy(200, scale=1e-3)

Sinc Profile
^^^^^^^^^^^^^

:func:`~torchoptics.profiles.sinc` — 2D sinc function (far-field diffraction of a rectangle):

.. code-block:: python

    from torchoptics.profiles import sinc

    s = sinc(200, scale=1e-3)

Siemens Star
^^^^^^^^^^^^^

:func:`~torchoptics.profiles.siemens_star` — Siemens star resolution target:

.. code-block:: python

    from torchoptics.profiles import siemens_star

    star = siemens_star(200, num_spokes=36, radius=1e-3)


Spatial Coherence Models
-------------------------

For spatially incoherent or partially coherent light, see :ref:`user-guide-spatial-coherence`.

:func:`~torchoptics.profiles.gaussian_schell_model` — Gaussian-Schell model coherence matrix:

.. code-block:: python

    from torchoptics.profiles import gaussian_schell_model

    gsm = gaussian_schell_model(30, waist_radius=40e-6, coherence_width=10e-6)

:func:`~torchoptics.profiles.schell_model` — General Schell model from arbitrary intensity and
coherence functions:

.. code-block:: python

    from torchoptics.profiles import schell_model

    def intensity_func(x, y):
        return torch.exp(-(x**2 + y**2) / (50e-6)**2)

    def coherence_func(dx, dy):
        return torch.exp(-(dx**2 + dy**2) / (20e-6)**2)

    sm = schell_model(30, intensity_func, coherence_func)
