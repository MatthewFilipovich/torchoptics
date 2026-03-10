.. _user-guide-polarization:

Polarization
=============

TorchOptics supports full vectorial simulation of polarized optical fields using Jones calculus.
Polarized fields carry three electric field components :math:`(E_x, E_y, E_z)`, and polarization
optical elements act on these components through :math:`3 \times 3` Jones matrices.

.. contents:: On This Page
   :local:
   :depth: 2

Polarized Fields
-----------------

Creating Polarized Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^

A polarized field is simply a :class:`~torchoptics.Field` whose data tensor has at least three
dimensions, where the third-to-last dimension has size 3 representing the polarization components
:math:`(E_x, E_y, E_z)`:

.. code-block:: python

    import torch
    from torchoptics import Field

    shape = 100

    # x-polarized field (horizontal)
    data = torch.zeros(3, shape, shape)
    data[0] = 1  # Set Ex component
    x_pol = Field(data).normalize()

    # y-polarized field (vertical)
    data = torch.zeros(3, shape, shape)
    data[1] = 1  # Set Ey component
    y_pol = Field(data).normalize()

    # 45-degree linear polarization (equal Ex and Ey)
    data = torch.zeros(3, shape, shape)
    data[0] = 1
    data[1] = 1
    diag_pol = Field(data).normalize()

    # Right-circular polarization
    data = torch.zeros(3, shape, shape, dtype=torch.complex64)
    data[0] = 1
    data[1] = 1j
    rcp = Field(data).normalize()

Visualizing Polarized Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When visualizing a polarized field, you typically select a specific polarization component:

.. code-block:: python

    # Visualize the x-component
    field.visualize(0, title="Ex component")

    # Visualize the y-component
    field.visualize(1, title="Ey component")

Splitting Polarization Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`~torchoptics.Field.polarized_split` method splits a polarized field into its three
component fields:

.. code-block:: python

    ex_field, ey_field, ez_field = polarized_field.polarized_split()


Polarization Elements
----------------------

Linear Polarizers
^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.LinearPolarizer` transmits the component of the field along its
transmission axis, attenuating the orthogonal component. The transmitted power follows Malus's law:

.. math::
    P = P_0 \cos^2(\theta - \theta_{\text{pol}})

where :math:`\theta` is the field's polarization angle and :math:`\theta_{\text{pol}}` is the
polarizer's transmission axis angle.

.. code-block:: python

    from torchoptics.elements import LinearPolarizer
    import torch

    # Create polarizers at different angles
    h_pol = LinearPolarizer(shape=100, theta=0, z=0.1)       # Horizontal
    v_pol = LinearPolarizer(shape=100, theta=torch.pi/2, z=0.1)  # Vertical
    d_pol = LinearPolarizer(shape=100, theta=torch.pi/4, z=0.1)  # 45 degrees

    # Apply to a polarized field
    output = h_pol(polarized_field)

Circular Polarizers
^^^^^^^^^^^^^^^^^^^^

:class:`~torchoptics.elements.LeftCircularPolarizer` and
:class:`~torchoptics.elements.RightCircularPolarizer` transmit only the corresponding circular
polarization component:

.. code-block:: python

    from torchoptics.elements import LeftCircularPolarizer, RightCircularPolarizer

    lcp = LeftCircularPolarizer(shape=100, z=0.1)
    rcp = RightCircularPolarizer(shape=100, z=0.1)

Waveplates
^^^^^^^^^^^

Waveplates introduce a relative phase delay between the fast and slow axes without absorbing light.

**Quarter-wave plate** (:math:`\varphi = \pi/2`): converts linear to circular polarization:

.. code-block:: python

    from torchoptics.elements import QuarterWaveplate

    # Fast axis at 45° — converts x-polarized to right-circular
    qwp = QuarterWaveplate(shape=100, theta=torch.pi/4, z=0.1)

**Half-wave plate** (:math:`\varphi = \pi`): rotates linear polarization:

.. code-block:: python

    from torchoptics.elements import HalfWaveplate

    hwp = HalfWaveplate(shape=100, theta=torch.pi/8, z=0.1)

**General waveplate**: arbitrary phase delay between axes:

.. code-block:: python

    from torchoptics.elements import Waveplate

    wp = Waveplate(shape=100, phi=torch.pi/3, theta=0, z=0.1)

Polarizing Beam Splitter
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~torchoptics.elements.PolarizingBeamSplitter` separates a polarized field into its
:math:`x` and :math:`y` components:

.. code-block:: python

    from torchoptics.elements import PolarizingBeamSplitter

    pbs = PolarizingBeamSplitter(shape=100, z=0.5)
    x_field, y_field = pbs(polarized_field)


Custom Polarization Modulators
-------------------------------

For arbitrary spatially varying polarization transformations, use the polarized modulator classes:

:class:`~torchoptics.elements.PolarizedModulator` — Full :math:`3 \times 3` complex Jones matrix
at each spatial point:

.. code-block:: python

    from torchoptics.elements import PolarizedModulator

    # (3, 3, H, W) complex tensor
    jones = torch.eye(3, dtype=torch.complex64).unsqueeze(-1).unsqueeze(-1)
    jones = jones.expand(3, 3, 100, 100).clone()
    modulator = PolarizedModulator(jones, z=0.1)

:class:`~torchoptics.elements.PolarizedPhaseModulator` — Phase-only Jones matrix:

.. code-block:: python

    from torchoptics.elements import PolarizedPhaseModulator

    phase = torch.zeros(3, 3, 100, 100)
    phase[0, 0] = torch.pi / 4  # Phase shift on Ex→Ex component
    pol_phase_mod = PolarizedPhaseModulator(phase, z=0.1)

:class:`~torchoptics.elements.PolarizedAmplitudeModulator` — Amplitude-only Jones matrix:

.. code-block:: python

    from torchoptics.elements import PolarizedAmplitudeModulator

    amplitude = torch.zeros(3, 3, 100, 100)
    amplitude[0, 0] = 1  # Pass Ex
    amplitude[1, 1] = 0.5  # Attenuate Ey by 50%
    pol_amp_mod = PolarizedAmplitudeModulator(amplitude, z=0.1)


Example: Malus's Law
----------------------

A classic demonstration of polarization: measuring the transmitted power through a rotating
linear polarizer as a function of angle.

.. code-block:: python

    import torch
    import torchoptics
    from torchoptics import Field
    from torchoptics.elements import LinearPolarizer

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    shape = 100

    # Create an x-polarized field
    data = torch.zeros(3, shape, shape)
    data[0] = 1
    field = Field(data).normalize()

    # Measure power through polarizers at different angles
    angles = torch.linspace(0, 2 * torch.pi, 100)
    powers = []
    for theta in angles:
        pol = LinearPolarizer(shape, theta=theta, z=0)
        output = pol(field)
        powers.append(output.power().item())

    # The result follows cos²(θ), i.e., Malus's law

Example: Generating Circular Polarization
-------------------------------------------

Use a quarter-wave plate at 45° to convert linearly polarized light to circular:

.. code-block:: python

    import torch
    import torchoptics
    from torchoptics import Field
    from torchoptics.elements import QuarterWaveplate

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    shape = 100

    # x-polarized input
    data = torch.zeros(3, shape, shape)
    data[0] = 1
    field = Field(data).normalize()

    # Quarter-wave plate with fast axis at 45°
    qwp = QuarterWaveplate(shape, theta=torch.pi / 4, z=0)
    circular = qwp(field)

    # The output has equal |Ex| and |Ey| with π/2 phase difference
    ex, ey, ez = circular.polarized_split()
    print(f"Ex power: {ex.power():.4f}")
    print(f"Ey power: {ey.power():.4f}")
