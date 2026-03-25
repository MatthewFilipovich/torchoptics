Spatial Coherence
=================


The :class:`~torchoptics.SpatialCoherence` class models partially coherent light through
the mutual coherence function, which is essential for simulating LEDs, thermal sources, and other
extended or broadband sources.


Mutual Coherence Function
--------------------------

Fully coherent light is described by a single field :math:`\psi(x,y)`. Partially coherent
light requires the 4D **mutual coherence function**:

.. math::

    \Gamma(x_1, y_1, x_2, y_2) = \langle \psi^*(x_1, y_1) \, \psi(x_2, y_2) \rangle

The diagonal :math:`\Gamma(x, y, x, y) = I(x, y)` gives the time-averaged intensity.


Creating a SpatialCoherence Object
-----------------------------------

:class:`~torchoptics.SpatialCoherence` takes a 4D complex tensor of shape ``(H, W, H, W)``.
The :mod:`torchoptics.profiles` module provides convenience functions for common models.

:func:`~torchoptics.profiles.gaussian_schell_model` — Gaussian intensity and coherence:

.. math::

    \Gamma(x_1, y_1, x_2, y_2) = \sqrt{I(x_1,y_1)\,I(x_2,y_2)} \cdot
    \exp\!\left(-\frac{|\mathbf{r}_1-\mathbf{r}_2|^2}{2\sigma_c^2}\right)

.. code-block:: python

    import torchoptics
    from torchoptics import SpatialCoherence
    from torchoptics.profiles import gaussian_schell_model

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    low_coh = SpatialCoherence(
        gaussian_schell_model(30, waist_radius=40e-6, coherence_width=10e-6)
    )
    high_coh = SpatialCoherence(
        gaussian_schell_model(30, waist_radius=40e-6, coherence_width=1e-3)
    )

:func:`~torchoptics.profiles.schell_model` — custom intensity and coherence functions:

.. code-block:: python

    from torchoptics.profiles import schell_model
    import torch

    data = schell_model(
        shape=30,
        intensity_func=lambda x, y: torch.exp(-2 * (x**2 + y**2) / (40e-6)**2),
        coherence_func=lambda dx, dy: torch.exp(-(dx**2 + dy**2) / (2 * (20e-6)**2)),
    )
    custom_coh = SpatialCoherence(data)


Propagation
-----------

Propagation applies the operator :math:`\Gamma' = U\,\Gamma\,U^\dagger` automatically using the
same API as coherent fields:

.. code-block:: python

    propagated = low_coh.propagate_to_z(0.01)

Low-coherence fields lose spatial structure during propagation, while high-coherence fields
maintain their distribution, matching the difference between LEDs and lasers.



Intensity and Power
-------------------

.. code-block:: python

    I = low_coh.intensity()     # Diagonal of Γ — shape (H, W)
    P = low_coh.power()         # Total power (scalar)


Modulation
----------

Scalar modulation transforms the coherence matrix:

.. math::

    \Gamma'(x_1, y_1, x_2, y_2) = \mathcal{M}^*(x_1, y_1) \cdot \Gamma \cdot \mathcal{M}(x_2, y_2)

.. code-block:: python

    from torchoptics.profiles import circle

    apertured = low_coh.modulate(circle(30, radius=100e-6))


Visualization
-------------

:meth:`~torchoptics.SpatialCoherence.visualize` displays the time-averaged intensity:

.. plot::
    :context: close-figs

    low_coh.visualize(title="Low Coherence")


Notes
-----

The ``(H, W, H, W)`` coherence matrix is memory-intensive; keep grid sizes small (typically
20–50 points per dimension) or use GPU acceleration.
