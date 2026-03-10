.. _user-guide-elements:

Optical Elements
=================

TorchOptics provides a comprehensive set of optical elements that model physical components such as
lenses, modulators, detectors, beam splitters, polarizers, and waveplates. All elements are
differentiable and can be composed into optical systems.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

All optical elements inherit from the :class:`~torchoptics.elements.Element` base class, which itself
extends :class:`~torchoptics.PlanarGrid`. Every element has:

- A **grid shape** defining its spatial extent
- A **z-position** along the optical axis
- A **spacing** and **offset** defining the physical grid
- A ``forward()`` method that takes a :class:`~torchoptics.Field` and returns a transformed result

Elements are used like PyTorch modules — you can call them directly on a field:

.. code-block:: python

    output_field = element(input_field)

Element Hierarchy
^^^^^^^^^^^^^^^^^^

The element class hierarchy provides specialized base classes for different types of transformations:

- :class:`~torchoptics.elements.Element` — Base class for all elements
- :class:`~torchoptics.elements.ModulationElement` — Elements that multiply the field by a spatial profile
- :class:`~torchoptics.elements.PolychromaticModulationElement` — Wavelength-dependent modulation
- :class:`~torchoptics.elements.PolarizedModulationElement` — Polarization-dependent modulation (Jones matrices)

Lenses
------

Thin Lens
^^^^^^^^^

The :class:`~torchoptics.elements.Lens` models a thin lens that applies a quadratic phase factor
and a circular aperture:

.. math::
    t(x, y) = \text{circ}\!\left(\frac{r}{R}\right) \exp\!\left(-\frac{i\pi(x^2 + y^2)}{\lambda f}\right)

where :math:`f` is the focal length, :math:`R` is the aperture radius, and :math:`r = \sqrt{x^2 + y^2}`.

.. code-block:: python

    from torchoptics.elements import Lens

    # Create a lens with 20 cm focal length at z = 0.3 m
    lens = Lens(shape=500, focal_length=0.2, z=0.3)

    # Apply to a field
    output = lens(input_field)

    # Visualize the lens profile
    lens.visualize(title="Thin Lens")

The focal length is a registered parameter and can be made learnable:

.. code-block:: python

    from torch.nn import Parameter
    import torch

    # Learnable focal length
    lens = Lens(shape=500, focal_length=Parameter(torch.tensor(0.2)), z=0.3)

Cylindrical Lens
^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.CylindricalLens` focuses light along a single axis, defined by
the orientation angle :math:`\theta`:

.. code-block:: python

    from torchoptics.elements import CylindricalLens

    # Horizontal cylindrical lens (focuses in y)
    cyl_lens = CylindricalLens(shape=500, focal_length=0.2, theta=0, z=0.3)

    # 45-degree oriented cylindrical lens
    cyl_lens_45 = CylindricalLens(shape=500, focal_length=0.2, theta=torch.pi / 4, z=0.3)

Modulators
----------

Modulators apply spatial amplitude and/or phase patterns to a field. They are the building blocks
for modeling spatial light modulators (SLMs), diffractive optical elements (DOEs), masks, and
apertures.

Complex Modulator
^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.Modulator` applies an arbitrary complex modulation profile:

.. code-block:: python

    from torchoptics.elements import Modulator

    # Create a complex modulation pattern
    modulation = torch.exp(1j * torch.randn(200, 200))
    modulator = Modulator(modulation, z=0.1)

Phase Modulator
^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.PhaseModulator` applies only a phase modulation :math:`e^{i\varphi(x,y)}`,
preserving the field amplitude:

.. code-block:: python

    from torchoptics.elements import PhaseModulator
    from torch.nn import Parameter

    # Fixed phase mask
    phase = torch.zeros(200, 200)
    phase_mod = PhaseModulator(phase, z=0.1)

    # Trainable phase modulator (for optimization)
    trainable_phase = Parameter(torch.zeros(200, 200))
    trainable_mod = PhaseModulator(trainable_phase, z=0.1)

Phase modulators are commonly used to model spatial light modulators and diffractive optical elements
in optimization problems.

Amplitude Modulator
^^^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.AmplitudeModulator` applies amplitude-only modulation, leaving
the phase unchanged:

.. code-block:: python

    from torchoptics.elements import AmplitudeModulator

    # Circular aperture as an amplitude modulator
    from torchoptics.profiles import circle
    aperture = circle(200, radius=1e-3)
    amp_mod = AmplitudeModulator(aperture, z=0.1)

Polychromatic Phase Modulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.PolychromaticPhaseModulator` applies wavelength-dependent phase
modulation based on an optical path length (OPL) profile:

.. math::
    t(x, y; \lambda) = \exp\!\left(\frac{2\pi i \cdot \text{OPL}(x,y)}{\lambda}\right)

.. code-block:: python

    from torchoptics.elements import PolychromaticPhaseModulator

    opl = torch.randn(200, 200) * 1e-6  # Optical path length profile
    poly_mod = PolychromaticPhaseModulator(opl, z=0.1)

Detectors
---------

Detector
^^^^^^^^

The :class:`~torchoptics.elements.Detector` measures the intensity of a field, returning a tensor
rather than a Field object:

.. code-block:: python

    from torchoptics.elements import Detector

    detector = Detector(shape=200, z=1.0)
    intensity = detector(field)  # Returns tensor of shape (..., H, W)

Linear Detector
^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.LinearDetector` computes weighted sums of the intensity using a
multi-channel weight matrix. This is useful for classification tasks where different spatial regions
correspond to different output classes:

.. code-block:: python

    from torchoptics.elements import LinearDetector

    # 10 output channels, each with a 200×200 weight pattern
    weight = torch.randn(10, 200, 200)
    lin_detector = LinearDetector(weight, z=1.0)

    # Returns tensor of shape (..., 10) — one value per channel
    output = lin_detector(field)

Beam Splitters
--------------

Beam Splitter
^^^^^^^^^^^^^

The :class:`~torchoptics.elements.BeamSplitter` models a 2×2 beam splitter with a unitary
transfer matrix parameterized by angles:

.. code-block:: python

    from torchoptics.elements import BeamSplitter
    import torch

    # 50:50 beam splitter
    bs = BeamSplitter(
        shape=200,
        theta=torch.pi / 4,  # 50:50 splitting ratio
        phi_0=0, phi_r=0, phi_t=0,
        z=0.5,
    )

    # Split a single field into two outputs
    reflected, transmitted = bs(field)

    # Combine two input fields
    output1, output2 = bs(field1, field2)

Polarizing Beam Splitter
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.PolarizingBeamSplitter` separates the orthogonal polarization
components of a polarized field:

.. code-block:: python

    from torchoptics.elements import PolarizingBeamSplitter

    pbs = PolarizingBeamSplitter(shape=200, z=0.5)
    field_x, field_y = pbs(polarized_field)


Polarizers
----------

Linear Polarizer
^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.LinearPolarizer` transmits light polarized along its transmission
axis at angle :math:`\theta`:

.. code-block:: python

    from torchoptics.elements import LinearPolarizer

    # Horizontal polarizer (θ = 0)
    h_pol = LinearPolarizer(shape=200, theta=0, z=0.1)

    # 45-degree polarizer
    pol_45 = LinearPolarizer(shape=200, theta=torch.pi / 4, z=0.1)

    # The transmitted power follows Malus's law: P = P₀ cos²(θ)
    output = h_pol(polarized_field)

Circular Polarizers
^^^^^^^^^^^^^^^^^^^^

:class:`~torchoptics.elements.LeftCircularPolarizer` and
:class:`~torchoptics.elements.RightCircularPolarizer` transmit left- or right-circularly
polarized light:

.. code-block:: python

    from torchoptics.elements import LeftCircularPolarizer, RightCircularPolarizer

    lcp = LeftCircularPolarizer(shape=200, z=0.1)
    rcp = RightCircularPolarizer(shape=200, z=0.1)


Waveplates
----------

Waveplates introduce a phase delay between the fast and slow axes of the field, changing the
polarization state without affecting the total power.

General Waveplate
^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.Waveplate` applies an arbitrary phase delay :math:`\varphi`
between the fast and slow axes oriented at angle :math:`\theta`:

.. code-block:: python

    from torchoptics.elements import Waveplate

    # Custom waveplate with π/3 phase delay, fast axis at 30°
    wp = Waveplate(shape=200, phi=torch.pi / 3, theta=torch.pi / 6, z=0.1)

Quarter-Wave Plate
^^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.QuarterWaveplate` has :math:`\varphi = \pi/2` and converts
linear polarization to circular polarization (and vice versa):

.. code-block:: python

    from torchoptics.elements import QuarterWaveplate

    qwp = QuarterWaveplate(shape=200, theta=torch.pi / 4, z=0.1)

Half-Wave Plate
^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.HalfWaveplate` has :math:`\varphi = \pi` and rotates the
polarization axis:

.. code-block:: python

    from torchoptics.elements import HalfWaveplate

    hwp = HalfWaveplate(shape=200, theta=torch.pi / 4, z=0.1)

Identity Element
-----------------

The :class:`~torchoptics.elements.IdentityElement` passes the field through unchanged. It is useful as
a placeholder or for defining measurement planes in a :class:`~torchoptics.System`:

.. code-block:: python

    from torchoptics.elements import IdentityElement

    # Mark a plane at z=0.5 for measurement
    marker = IdentityElement(shape=200, z=0.5)

Custom Elements
----------------

You can create custom optical elements by subclassing :class:`~torchoptics.elements.ModulationElement`
and implementing the ``modulation_profile()`` method:

.. code-block:: python

    from torchoptics.elements import ModulationElement

    class GaussianAperture(ModulationElement):
        def __init__(self, shape, waist, z=0, spacing=None, offset=None):
            super().__init__(shape, z, spacing, offset)
            self.register_optics_property("waist", waist, is_positive=True)

        def modulation_profile(self):
            y, x = self.meshgrid()
            return torch.exp(-(x**2 + y**2) / self.waist**2)

    # Use like any built-in element
    aperture = GaussianAperture(shape=200, waist=1e-3, z=0.1)
    output = aperture(field)

For more about combining elements into systems, see :ref:`user-guide-systems`.
