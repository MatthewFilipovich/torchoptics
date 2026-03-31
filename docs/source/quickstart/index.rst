Quickstart
==========

This guide introduces the core workflow of TorchOptics: creating optical fields, propagating them
through free space, simulating optical elements, building multi-element systems, and optimizing
optical designs using gradient descent.

Before starting, make sure TorchOptics is installed (:ref:`installation`).


Overview
--------

TorchOptics simulates optical systems using Fourier optics, where light is modeled as complex-valued
wavefronts sampled on 2D grids. Built on PyTorch, every operation is
fully differentiable, enabling gradient-based optimization of optical designs.

The library is built around three core abstractions:

- :class:`~torchoptics.Field` — A monochromatic optical field: complex-valued data on a 2D planar
  grid at a position along the optical axis.
- :class:`~torchoptics.elements.Element` (e.g., :class:`~torchoptics.elements.Lens`,
  :class:`~torchoptics.elements.PhaseModulator`) — Optical components that transform fields
  at specified :math:`z` positions.
- :class:`~torchoptics.System` — An ordered sequence of elements forming a complete optical setup,
  analogous to :class:`torch.nn.Sequential`.

Two global defaults set the physical scale of the simulation:

- **Spacing** — Physical distance between adjacent grid points (meters).
- **Wavelength** — Wavelength of the monochromatic light (meters).


Setup
------

Import TorchOptics and set the two global defaults (grid spacing and wavelength) that all
subsequent fields and elements will inherit:

.. plot::
    :context: reset

    import torch
    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import AmplitudeModulator, Lens
    from torchoptics.profiles import checkerboard, circle, gaussian

    torchoptics.set_default_spacing(10e-6)       # 10 µm grid spacing
    torchoptics.set_default_wavelength(700e-9)   # 700 nm (red light)


Fields and Propagation
-----------------------

A :class:`~torchoptics.Field` wraps a 2D complex-valued tensor together with its spatial geometry
(grid shape, spacing, offset, and :math:`z` position). The :mod:`torchoptics.profiles` module
provides functions for common spatial profiles (Gaussian beams, geometric apertures, gratings, etc.).

Let's create a field from a circular aperture:

.. plot::
    :context: close-figs

    shape = 1000  # 1000×1000 grid (10 mm × 10 mm physical extent)
    field = Field(circle(shape, radius=2e-3))
    field.visualize(title="Circular Aperture (z = 0)")

Use :meth:`~torchoptics.Field.propagate_to_z` to propagate a field through free space. As light
travels, it diffracts, producing characteristic patterns at different distances:

.. plot::
    :context: close-figs

    # Near-field (Fresnel) diffraction
    field.propagate_to_z(0.5).visualize(title="z = 0.5 m  (Fresnel region)")

.. plot::
    :context: close-figs

    # Far-field (Fraunhofer) diffraction: the Airy pattern
    field.propagate_to_z(10.0).visualize(title="z = 10.0 m  (Fraunhofer region)")

Close to the aperture (the Fresnel region), diffraction produces fringes near the edges. Far away
(the Fraunhofer region), the wavefront converges to the `Airy pattern
<https://en.wikipedia.org/wiki/Airy_disk>`_, the Fourier transform of the circular aperture.

.. note::

    TorchOptics automatically selects between the **angular spectrum method** (ASM) and the
    **direct integration method** (DIM) based on the propagation distance and grid geometry.
    This can be overridden with the ``propagation_method`` parameter.


Focusing with a Lens
---------------------

The :class:`~torchoptics.elements.Lens` models a thin lens with focal length :math:`f`. It applies
a quadratic phase factor with a circular aperture to the incident field:

.. math::

    \mathcal{M}(x, y) = \operatorname{circ}\!\left(\frac{r}{R}\right) \cdot
    \exp\!\left(-i \frac{\pi}{\lambda f}(x^2 + y^2)\right)

where :math:`r = \sqrt{x^2 + y^2}`, :math:`R` is the aperture radius (half the lens's
physical extent), :math:`\lambda` is the wavelength, and :math:`f` is the focal length.

Calling an element on a field (``lens(field)``) applies this transformation. Let's focus a Gaussian
beam with a 400 mm lens:

.. plot::
    :context: close-figs

    gaussian_beam = Field(gaussian(shape, waist_radius=3e-3))
    gaussian_beam.visualize(title="Gaussian Beam (z = 0)")

.. plot::
    :context: close-figs

    f = 1  # Focal length: 1 m
    lens = Lens(shape, f, z=0)

    focused = lens(gaussian_beam).propagate_to_z(f)
    focused.visualize(title=f"Focal Plane (z = {f} m)")

The beam converges to a tight spot at the focal plane.


Optical Systems
----------------

For multi-element setups, the :class:`~torchoptics.System` class handles propagation between
components automatically; you just specify where each element sits along the optical axis.
Use :meth:`~torchoptics.System.measure_at_z` to compute the field at any :math:`z` position.

As an example, let's build a `4f system
<https://en.wikipedia.org/wiki/Fourier_optics#4F_Correlator>`_: two lenses separated by
:math:`2f` with a spatial filter at the Fourier plane (:math:`z = 2f`). The system relays the
input image to the output plane (:math:`z = 4f`), while the Fourier plane in between gives direct
access to the spatial frequency content for filtering. Here we place a high-pass filter that
blocks low spatial frequencies, extracting edges from a checkerboard:

.. plot::
    :context: close-figs

    input_field = Field(checkerboard(shape, tile_length=400e-6, num_tiles=15))
    input_field.visualize(title="Input Field", vmax=1)

.. plot::
    :context: close-figs

    # High-pass filter at the Fourier plane (z = 2f)
    f = 200e-3  # Focal length: 200 mm
    filter_mask = 1 - circle(shape, radius=500e-6)
    aperture = AmplitudeModulator(filter_mask, z=2 * f)

    aperture.visualize(title="High-Pass Filter at Fourier Plane")

The :class:`~torchoptics.elements.AmplitudeModulator` at the Fourier plane applies a
transmittance mask: ``1 - circle(...)`` blocks the DC and low-frequency components, passing only
the high-frequency content.

.. plot::
    :context: close-figs

    # 4f system with high-pass filter at the Fourier plane
    system = System(
        Lens(shape, f, z=1 * f),
        aperture,
        Lens(shape, f, z=3 * f),
    )

    output = system.measure_at_z(input_field, z=4 * f)
    output.visualize(title="Output: Edges", vmax=1)

GPU Acceleration
-----------------

All TorchOptics objects are standard PyTorch modules. Move fields and systems to the GPU with
``.to()`` for accelerated computation, especially beneficial for large grids and optimization
loops:

.. code-block:: python

    device = "cuda" if torch.cuda.is_available() else "cpu"

    field = field.to(device)
    system = system.to(device)

    output = system.measure_at_z(field, z=4 * f)


Inverse Design
---------------

Every operation in TorchOptics is fully differentiable through :mod:`torch.autograd`. This means
you can optimize optical designs using gradient descent, the same approach used to train neural
networks.

As an example, let's train a diffractive system to reshape a Gaussian beam into an eight-petal
beam. First, we define the input and target fields:

.. plot::
    :context: reset

    import torch
    from torch.nn import Parameter
    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import PhaseModulator
    from torchoptics.profiles import gaussian, laguerre_gaussian

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    shape = 250
    waist_radius = 300e-6

    # Input: Gaussian beam
    input_field = Field(gaussian(shape, waist_radius=waist_radius), z=0)
    input_field.visualize(title="Input: Gaussian")

The target is a superposition of two Laguerre-Gaussian modes with opposite orbital angular momentum:

.. math::

    \psi_\text{target} = \mathrm{LG}_0^{+4} + \mathrm{LG}_0^{-4}

whose interference produces an eight-petal intensity pattern.

.. plot::
    :context: close-figs

    # Target: eight-petal beam (LG_0^{+4} + LG_0^{-4} superposition)
    target_data = laguerre_gaussian(shape, p=0, l=4, waist_radius=waist_radius) \
                + laguerre_gaussian(shape, p=0, l=-4, waist_radius=waist_radius)
    target_field = Field(target_data, z=0.8).normalize()  # normalize to unit power
    target_field.visualize(title="Target: Petal Beam")

The loss is :math:`1 - |\eta|^2`, where :math:`\eta` is the inner product (mode overlap) between the
output and target fields: equal to 1 when they are identical and 0 when orthogonal.

.. plot::
    :context: close-figs

    # Trainable diffractive system: three phase planes initialized to zero
    system = System(
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.2),
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.4),
        PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.6),
    )

    optimizer = torch.optim.Adam(system.parameters(), lr=0.05)
    for iteration in range(200):
        optimizer.zero_grad()
        output = system.measure_at_z(input_field, z=0.8)
        loss = 1 - output.inner(target_field).abs().square()  # 1 - |η|²
        loss.backward()
        optimizer.step()

.. plot::
    :context: close-figs

    with torch.no_grad():
        result = system.measure_at_z(input_field, z=0.8)
    result.visualize(title="Optimized Output")

The optimizer discovers the phase patterns that reshape the beam into the target distribution,
with no manual optical design required. This approach scales to complex objectives involving multiple
elements, custom loss functions, and joint optimization with neural networks.

.. tip::

    See the :doc:`optimization examples </examples/optimization/index>` for complete inverse design
    workflows with loss curves and animations.


Next Steps
-----------

- :doc:`/user-guide/index` — In-depth guides on fields, elements, and systems.
- :doc:`/examples/index` — Diffraction, polarization, spatial coherence, and inverse design examples.
- :doc:`/api-reference/index` — Complete API documentation.
