.. _user-guide-spatial-coherence:

Spatial Coherence
==================

TorchOptics supports the simulation of spatially incoherent and partially coherent light through
the :class:`~torchoptics.SpatialCoherence` class. This allows modeling of extended sources, LEDs,
and other light sources where spatial coherence plays a significant role.

.. contents:: On This Page
   :local:
   :depth: 2

Background
-----------

Fully coherent light is described by a single complex field :math:`U(x, y)`. Partially coherent
light, however, requires a **mutual coherence function** (or cross-spectral density):

.. math::
    \Gamma(x_1, y_1, x_2, y_2) = \langle U^*(x_1, y_1) \, U(x_2, y_2) \rangle

This 4D function describes the correlation between the field at two different spatial points. The
diagonal :math:`\Gamma(x, y, x, y) = \langle |U(x, y)|^2 \rangle` gives the average intensity.

Key concepts:

- **Fully coherent**: :math:`\Gamma` factorizes as :math:`U^*(x_1, y_1) \, U(x_2, y_2)`
- **Fully incoherent**: :math:`\Gamma(x_1, y_1, x_2, y_2) = I(x_1, y_1) \, \delta(x_1 - x_2, y_1 - y_2)`
- **Partially coherent**: :math:`\Gamma` has finite width — described by models like Gaussian-Schell

The SpatialCoherence Class
----------------------------

The :class:`~torchoptics.SpatialCoherence` class extends :class:`~torchoptics.Field` to work with
4D coherence matrices of shape ``(..., H, W, H, W)``:

.. code-block:: python

    from torchoptics import SpatialCoherence
    from torchoptics.profiles import gaussian_schell_model

    # Create a Gaussian-Schell model beam
    coherence_data = gaussian_schell_model(
        shape=30,
        waist_radius=40e-6,
        coherence_width=10e-6,
    )
    sc = SpatialCoherence(coherence_data)

Properties
^^^^^^^^^^^

:class:`~torchoptics.SpatialCoherence` shares the same API as :class:`~torchoptics.Field` but
operates on the coherence matrix:

.. code-block:: python

    # Intensity: diagonal of the coherence matrix
    intensity = sc.intensity()  # Shape: (..., H, W)

    # Total power
    power = sc.power()

    # Normalization
    sc_normalized = sc.normalize(normalized_power=1.0)


Creating Coherence Matrices
-----------------------------

Gaussian-Schell Model
^^^^^^^^^^^^^^^^^^^^^^

The most commonly used model for partially coherent light is the **Gaussian-Schell model**, where
both the intensity and coherence are Gaussian:

.. math::
    \Gamma(x_1, y_1, x_2, y_2) = \sqrt{I(x_1, y_1)} \sqrt{I(x_2, y_2)} \, \mu(\Delta x, \Delta y)

with Gaussian intensity :math:`I(x, y) = \exp\!\left(-\frac{2(x^2 + y^2)}{w^2}\right)` and
Gaussian coherence :math:`\mu(\Delta x, \Delta y) = \exp\!\left(-\frac{\Delta x^2 + \Delta y^2}{2\sigma_c^2}\right)`.

.. code-block:: python

    from torchoptics.profiles import gaussian_schell_model

    # High coherence (large coherence width)
    high_coh = gaussian_schell_model(30, waist_radius=40e-6, coherence_width=1e-3)

    # Low coherence (small coherence width)
    low_coh = gaussian_schell_model(30, waist_radius=40e-6, coherence_width=10e-6)

The ``coherence_width`` parameter :math:`\sigma_c` controls the degree of spatial coherence:

- **Large** :math:`\sigma_c \gg w`: Approaches fully coherent — the field maintains its
  spatial structure during propagation
- **Small** :math:`\sigma_c \ll w`: Approaches fully incoherent — the field loses its spatial
  structure rapidly

General Schell Model
^^^^^^^^^^^^^^^^^^^^^

For custom intensity and coherence distributions, use the general
:func:`~torchoptics.profiles.schell_model`:

.. code-block:: python

    import torch
    from torchoptics.profiles import schell_model

    def intensity_func(x, y):
        return torch.exp(-(x**2 + y**2) / (50e-6)**2)

    def coherence_func(dx, dy):
        return torch.exp(-(dx**2 + dy**2) / (20e-6)**2)

    custom_coherence = schell_model(30, intensity_func, coherence_func)

From Field Outer Products
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can construct coherence matrices from fields using the outer product:

.. code-block:: python

    from torchoptics import Field
    from torchoptics.profiles import gaussian

    field = Field(gaussian(30, waist_radius=40e-6))
    coherence_matrix = field.outer(field)  # Fully coherent

    sc = SpatialCoherence(coherence_matrix)


Propagation of Partially Coherent Light
-----------------------------------------

Partially coherent light propagates according to:

.. math::
    \Gamma'(x_1, y_1, x_2, y_2) = \iint \iint h^*(x_1 - x_1', y_1 - y_1') \, h(x_2 - x_2', y_2 - y_2')
    \, \Gamma(x_1', y_1', x_2', y_2') \, dx_1' \, dy_1' \, dx_2' \, dy_2'

or in matrix form: :math:`\Gamma' = U \Gamma U^\dagger`, where :math:`U` is the propagation operator.

TorchOptics handles this automatically — you use the same propagation methods as for coherent fields:

.. code-block:: python

    from torchoptics import SpatialCoherence
    from torchoptics.profiles import gaussian_schell_model

    sc = SpatialCoherence(gaussian_schell_model(30, 40e-6, 10e-6))

    # Propagate to z = 0.01 m
    propagated = sc.propagate_to_z(0.01)

    # Visualize the intensity
    propagated.visualize(title="Propagated Partially Coherent Field")


Modulation of Partially Coherent Light
---------------------------------------

When a partially coherent field passes through a modulation element, the coherence matrix is
transformed as :math:`\Gamma' = t^* \, \Gamma \, t^T`, where :math:`t` is the element's modulation
profile. This is handled automatically:

.. code-block:: python

    from torchoptics.elements import Lens

    lens = Lens(shape=30, focal_length=0.1, z=0.01)
    modulated = lens(sc.propagate_to_z(0.01))


Example: Coherence Effects on Propagation
-------------------------------------------

This example demonstrates how coherence width affects the propagation behavior of partially
coherent beams:

.. code-block:: python

    import torchoptics
    from torchoptics import SpatialCoherence
    from torchoptics.profiles import gaussian_schell_model

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    shape = 30
    waist_radius = 40e-6

    # Low coherence: field washes out during propagation
    low_coh = SpatialCoherence(
        gaussian_schell_model(shape, waist_radius, coherence_width=10e-6)
    )

    # High coherence: field maintains structure
    high_coh = SpatialCoherence(
        gaussian_schell_model(shape, waist_radius, coherence_width=1e-3)
    )

    # Propagate both and compare
    for z in [0, 0.01, 0.02]:
        low_coh.propagate_to_z(z).visualize(
            title=f"Low Coherence, z={z} m", vmin=0
        )
        high_coh.propagate_to_z(z).visualize(
            title=f"High Coherence, z={z} m", vmin=0
        )

.. note::
    Spatial coherence simulations use 4D coherence matrices, so the grid size must be kept small
    (typically 20–50 points per dimension) to be computationally feasible. The memory requirement
    scales as :math:`O(N^4)`.
