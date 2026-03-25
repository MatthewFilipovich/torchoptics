Elements
========


Optical elements are planar components that transform fields at specific positions along the
optical axis. All elements inherit from :class:`~torchoptics.elements.Element` and share the
same geometry properties as :class:`~torchoptics.Field` (``shape``, ``spacing``, ``offset``,
``z``).


Using Elements
--------------

Call an element on a field to apply its transformation. The element validates that the field's
geometry matches, applies its operation, and returns the result:

.. code-block:: python

    output_field = element(input_field)

Elements are PyTorch modules (:class:`torch.nn.Module`); they can be moved to the GPU,
serialized, and composed into :class:`~torchoptics.System` objects.


Modulators
----------

Modulators apply a point-wise complex multiplication:

.. math::

    \psi'(x, y) = \mathcal{M}(x, y) \cdot \psi(x, y)

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Element
     - Profile
   * - :class:`~torchoptics.elements.Modulator`
     - Arbitrary complex: :math:`\mathcal{M} = m(x,y)`.
   * - :class:`~torchoptics.elements.PhaseModulator`
     - Phase-only: :math:`\mathcal{M} = e^{i\phi(x,y)}`.
   * - :class:`~torchoptics.elements.AmplitudeModulator`
     - Amplitude-only: :math:`\mathcal{M} = a(x,y)`.
   * - :class:`~torchoptics.elements.PolychromaticPhaseModulator`
     - Wavelength-dependent phase: :math:`\mathcal{M} = e^{i\,2\pi\,(n(\lambda)-1)\,t/\lambda}`,
       where :math:`t` is the physical thickness and :math:`n(\lambda)` is the refractive index.
   * - :class:`~torchoptics.elements.IdentityElement`
     - Pass-through: :math:`\mathcal{M} = 1` (useful as a placeholder in systems).

.. plot::
    :context: reset

    import torch
    import torchoptics
    from torchoptics import Field
    from torchoptics.elements import PhaseModulator, AmplitudeModulator
    from torchoptics.profiles import gaussian, circle

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)
    shape = 300

    amp_mod = AmplitudeModulator(circle(shape, radius=1e-3), z=0)
    beam = Field(gaussian(shape, waist_radius=1.5e-3))
    amp_mod(beam).visualize(title="After Amplitude Modulator")

.. plot::
    :context: close-figs

    torch.manual_seed(0)
    phase_mod = PhaseModulator(torch.randn(shape, shape), z=0)
    phase_mod(beam).visualize(title="After Phase Modulator")

To make a modulator learnable, wrap its data in :class:`torch.nn.Parameter`:

.. code-block:: python

    from torch.nn import Parameter

    trainable_phase = PhaseModulator(Parameter(torch.zeros(300, 300)), z=0)

See :doc:`inverse_design` for full details on trainable properties.

:class:`~torchoptics.elements.PolychromaticPhaseModulator` takes a physical thickness profile
and a refractive index (scalar or wavelength-dependent callable), computing the phase at run
time; the same physical element produces different phase shifts for different wavelengths:

.. code-block:: python

    from torchoptics.elements import PolychromaticPhaseModulator
    from torchoptics.profiles import blazed_grating

    # A blazed grating as a polychromatic element (constant n=1.5)
    thickness = blazed_grating(300, period=100e-6, height=700e-9)
    grating = PolychromaticPhaseModulator(thickness, n=1.5, z=0)

    # Or use a dispersive medium with a callable refractive index
    def sellmeier(wl):
        return 1.5 + 0.01e-12 / wl**2  # simplified example

    grating_dispersive = PolychromaticPhaseModulator(thickness, n=sellmeier, z=0)

    # Call with a field that carries its own wavelength
    output = grating(field)


Lenses
------

:class:`~torchoptics.elements.Lens` models a thin lens with focal length :math:`f`, applying a
quadratic phase within a circular aperture:

.. math::

    \mathcal{M}(x, y) = \operatorname{circ}(r) \cdot
    \exp\!\left(-i \frac{\pi}{\lambda f}(x^2 + y^2)\right)

The phase is wavelength-dependent, matching real lens behavior.

.. plot::
    :context: close-figs

    from torchoptics.elements import Lens

    lens = Lens(shape, 200e-3, z=0)
    lens.visualize(title="Thin Lens Profile")

:class:`~torchoptics.elements.CylindricalLens` focuses along a single axis at orientation angle
:math:`\theta`:

.. code-block:: python

    from torchoptics.elements import CylindricalLens

    cyl_lens = CylindricalLens(shape, focal_length=100e-3, theta=0, z=0)


Detectors
---------

Detectors convert fields into intensity measurements, returning **tensors** rather than fields.

:class:`~torchoptics.elements.Detector` returns the power per grid cell
:math:`P_{i,j} = I_{i,j} \cdot \Delta A`:

.. code-block:: python

    from torchoptics.elements import Detector

    detector = Detector(shape, z=0.5)
    power_map = detector(field)  # Returns a Tensor

:class:`~torchoptics.elements.LinearDetector` applies a (C, H, W) weight tensor and integrates,
producing C output channels, analogous to :class:`torch.nn.Linear`:

.. code-block:: python

    from torchoptics.elements import LinearDetector

    weight = torch.randn(10, 300, 300)
    lin_detector = LinearDetector(weight, z=0.5)
    outputs = lin_detector(field)  # Shape: (10,)


Beam Splitters
--------------

:class:`~torchoptics.elements.BeamSplitter` splits a field into two outputs via a 2×2 transfer
matrix parameterized by angles and phases:

.. code-block:: python

    import math
    from torchoptics.elements import BeamSplitter

    bs = BeamSplitter(shape, theta=math.pi/4, phi_0=0, phi_r=0, phi_t=0, z=0)
    output_1, output_2 = bs(field)

:class:`~torchoptics.elements.PolarizingBeamSplitter` splits by polarization component
(see :doc:`polarization`):

.. code-block:: python

    from torchoptics.elements import PolarizingBeamSplitter

    pbs = PolarizingBeamSplitter(shape, z=0)
    field_x, field_y = pbs(polarized_field)


Visualization
-------------

Most elements support :meth:`visualize` to display their modulation profile:

.. code-block:: python

    lens.visualize()              # Magnitude and phase
    polarizer.visualize(0, 0)    # Jones matrix component J[0,0]


.. _custom-elements:

Custom Elements
---------------

Subclass the appropriate base class and implement ``modulation_profile()``:

.. code-block:: python

    from torchoptics.elements import ModulationElement

    class VortexPlate(ModulationElement):
        """Applies azimuthal phase exp(i * l * phi)."""

        def __init__(self, shape, l, z=0, **kwargs):
            super().__init__(shape, z, **kwargs)
            self.l = l

        def modulation_profile(self):
            x, y = self.meshgrid()
            phi = torch.atan2(y, x)
            return torch.exp(1j * self.l * phi)

    vortex = VortexPlate(300, l=3, z=0)
    vortex(beam).visualize(title="After Vortex Plate (l=3)")

The base classes define the transformation patterns:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Base Class
     - Override
   * - :class:`~torchoptics.elements.ModulationElement`
     - ``modulation_profile()`` → complex tensor.
   * - :class:`~torchoptics.elements.PolychromaticModulationElement`
     - ``modulation_profile(wavelength)`` → wavelength-dependent.
   * - :class:`~torchoptics.elements.PolarizedModulationElement`
     - ``polarized_modulation_profile()`` → 3×3 Jones matrix.



