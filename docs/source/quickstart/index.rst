Quickstart
==========

Welcome to the TorchOptics quickstart guide! This tutorial walks you through the core concepts
of wave optics simulations with TorchOptics:

- Setting up the simulation environment with global defaults
- Creating optical fields using the :class:`~torchoptics.Field` class
- Propagating fields through free space
- Modeling lenses and imaging systems
- Building multi-element systems with the :class:`~torchoptics.System` class
- Using spatial profiles to define field distributions

Before starting, make sure TorchOptics is installed (:ref:`installation`).

.. contents:: On This Page
   :local:
   :depth: 2

Import TorchOptics
------------------

We first import the TorchOptics components we'll use:

.. plot::
    :context: reset

    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import Lens
    from torchoptics.profiles import triangle

Setting Up the Simulation
--------------------------

Every TorchOptics simulation requires two fundamental parameters:

- **Spacing**: The physical distance between adjacent grid points (in meters), which determines the spatial
  resolution of the simulation. Smaller spacing means higher resolution but more computation.
- **Wavelength**: The wavelength of the monochromatic light being simulated (in meters).

These can be set globally so you don't have to pass them to every function:

.. plot::
    :context:

    torchoptics.set_default_spacing(10e-6)  # 10 μm spacing
    torchoptics.set_default_wavelength(700e-9)  # 700 nm (red light)

.. tip::
    You can also specify ``spacing`` and ``wavelength`` explicitly when creating individual fields and
    elements, which will override the defaults. This is useful when different parts of your simulation
    use different parameters.

Creating an Optical Field
--------------------------

Optical fields in TorchOptics are represented by the :class:`~torchoptics.Field` class, which stores
complex-valued wavefronts sampled on a 2D grid in the :math:`xy`-plane. Each field has an associated
position along the optical axis (:math:`z`), grid spacing, and wavelength.

Let's create a field with a triangular amplitude profile using the :func:`~torchoptics.profiles.triangle`
profile function:

.. plot::
    :context: close-figs

    shape = 500  # Grid shape (500×500 points)
    base = 2e-3  # Triangle base width (2 mm)
    height = 1e-3  # Triangle height (1 mm)

    # Generate a triangular profile and create the field
    triangle_profile = triangle(shape, base, height)
    field = Field(triangle_profile)

    # Visualize the field
    field.visualize(title="Initial Field at z=0 m")
    print(field)

The :meth:`~torchoptics.Field.visualize` method displays the magnitude squared (intensity) and phase of the
complex field. Since this field has uniform phase, only the amplitude pattern is visible.

.. note::
    The ``shape`` parameter can be either a single integer ``N`` (for an ``N×N`` grid) or a tuple
    ``(H, W)`` for rectangular grids. For 1D simulations, set one dimension to 1, e.g., ``shape=(1, 500)``.

Free-Space Propagation
-----------------------

One of the central operations in wave optics is free-space propagation — computing how a field evolves as
it travels through space. TorchOptics implements this using Fourier optics methods:

- **Angular Spectrum Method (ASM)**: Uses FFT-based transfer functions. Efficient for large propagation
  distances and preserves the grid geometry.
- **Direct Integration Method (DIM)**: Numerically evaluates the Rayleigh-Sommerfeld integral. More
  flexible for short distances and different input/output geometries.

By default, TorchOptics automatically selects the best method (``propagation_method="AUTO"``).

Let's propagate our triangular field to :math:`z = 0.1` m and observe the diffraction pattern:

.. plot::
    :context: close-figs

    propagated_field = field.propagate_to_z(0.1)
    propagated_field.visualize(title="Propagated Field at z=0.1 m")

The sharp edges of the triangle have diffracted, producing the characteristic Fresnel diffraction
pattern. Note how the phase now shows spatial variation due to the propagation.

Image Formation with a Lens
-----------------------------

Lenses are fundamental optical elements that focus and image light. In TorchOptics, a thin lens is modeled
by the :class:`~torchoptics.elements.Lens` class, which applies a quadratic phase factor to the field's wavefront.

The thin lens equation relates the object distance :math:`d_o` and image distance :math:`d_i` to the focal
length :math:`f`:

.. math::
    \frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}

For this example, we choose :math:`f = 0.2` m and equal conjugate distances :math:`d_o = d_i = 0.4` m,
which produces a 1:1 image (unit magnification):

.. plot::
    :context:

    focal_length = 0.2  # Lens focal length (20 cm)
    d_o = 0.4  # Object-to-lens distance (40 cm)
    d_i = 0.4  # Lens-to-image distance (40 cm)

    lens_z = d_o  # Position of the lens along the z-axis
    image_z = d_o + d_i  # Position of the image plane along the z-axis

    print(f"Lens Position: {lens_z} m")
    print(f"Image Plane Position: {image_z} m")

Initialize the Lens
^^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.Lens` class takes a grid shape, focal length, and z-position. It also
applies a circular aperture by default, matching the physical extent of the grid:

.. plot::
    :context: close-figs

    lens = Lens(shape, focal_length, lens_z)
    lens.visualize(title="Lens Profile")
    print(lens)

The visualization shows the lens modulation profile: a circular aperture with a quadratic phase that
increases radially from the center.

Field at the Lens Plane
^^^^^^^^^^^^^^^^^^^^^^^^

We propagate the field from the object plane (:math:`z=0`) to the lens position:

.. plot::
    :context: close-figs

    field_before_lens = field.propagate_to_z(lens_z)
    field_before_lens.visualize(title="Field Before Lens")

Now we apply the lens transformation. In TorchOptics, elements act as callable functions — you simply
pass a field through them:

.. plot::
    :context: close-figs

    field_after_lens = lens(field_before_lens)
    field_after_lens.visualize(title="Field After Lens")

Notice how the lens has imparted a quadratic phase to the field, which will cause it to converge
at the image plane.

Field at the Image Plane
^^^^^^^^^^^^^^^^^^^^^^^^

We propagate from the lens to the image plane to obtain the final image:

.. plot::
    :context: close-figs

    field_image_plane = field_after_lens.propagate_to_z(image_z)
    field_image_plane.visualize(title="Field at Image Plane")

The inverted triangle at the image plane demonstrates the imaging property of the thin lens. The image
is inverted (rotated 180°) with unit magnification, consistent with geometric optics.

Building Systems
-----------------

Manually propagating a field to each element and applying it can become tedious for complex setups. The
:class:`~torchoptics.System` class simplifies this by representing an optical system as an ordered
sequence of elements — analogous to :class:`torch.nn.Sequential` in PyTorch.

When you call a system's :meth:`~torchoptics.System.forward` method (or call the system directly), it
automatically propagates the field between elements in order of their :math:`z`-positions.

Let's create a system with our lens:

.. plot::
    :context:

    system = System(lens)
    print(system)

Measuring at the Image Plane
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`~torchoptics.System.measure_at_z` method propagates a field through the system and then
continues propagation to any specified :math:`z`-position:

.. plot::
    :context: close-figs

    field_image_plane = system.measure_at_z(field, z=image_z)
    field_image_plane.visualize(title="Field at Image Plane (System)")

This produces the same result as the manual step-by-step approach above, but in a single line of code.

Next Steps
----------

Now that you understand the basics, explore the :doc:`/user-guide/index` for deeper coverage of:

- :doc:`/user-guide/fields` — Field properties, operations, and batch processing
- :doc:`/user-guide/elements` — All optical elements and their parameters
- :doc:`/user-guide/propagation` — Propagation methods and when to use each
- :doc:`/user-guide/profiles` — Spatial profiles for field initialization
- :doc:`/user-guide/optimization` — Training optical systems with gradient descent

Or jump to the :doc:`/examples/index` gallery for practical demonstrations.
