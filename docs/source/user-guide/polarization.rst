Polarization
============


TorchOptics represents polarized light using a three-component field. The polarization
dimension appears as the third-to-last dimension of the data tensor, with components
corresponding to the :math:`x`, :math:`y`, and :math:`z` polarization directions.


Polarized Fields
----------------

A polarized field has data of shape ``(3, H, W)``. For paraxial optics, the :math:`z` component
is typically zero:

.. plot::
    :context: reset

    import torch
    import torchoptics
    from torchoptics import Field

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)
    shape = 200

    data = torch.zeros(3, shape, shape, dtype=torch.cdouble)
    data[0] = 1.0  # x-component only
    x_pol = Field(data)
    x_pol.visualize(0, title="x-Polarized Field (x-component)", vmin=0, vmax=1)
    x_pol.visualize(1, title="x-Polarized Field (y-component)", vmin=0, vmax=1)
    x_pol.visualize(2, title="x-Polarized Field (z-component)", vmin=0, vmax=1)


Jones Calculus
--------------

Polarized elements apply extended 3×3 Jones matrices at each spatial point:

.. math::

    \begin{bmatrix} \psi'_x \\ \psi'_y \\ \psi'_z \end{bmatrix} =
    \mathbf{J}(x,y) \cdot
    \begin{bmatrix} \psi_x \\ \psi_y \\ \psi_z \end{bmatrix}

Each :math:`J_{ij}` is a 2D tensor (H × W), giving a full profile of shape ``(3, 3, H, W)``.


Polarizers
----------

:class:`~torchoptics.elements.LinearPolarizer` transmits a linear polarization component at
angle :math:`\theta`:

.. plot::
    :context: close-figs

    from torchoptics.elements import LinearPolarizer

    pol = LinearPolarizer(shape, theta=torch.pi/4, z=0)
    output = pol(x_pol)
    output.visualize(0, title="After 45° Polarizer (x-component)")

:class:`~torchoptics.elements.LeftCircularPolarizer` and
:class:`~torchoptics.elements.RightCircularPolarizer` project onto circular polarization states:

.. code-block:: python

    from torchoptics.elements import LeftCircularPolarizer, RightCircularPolarizer

    left_field = LeftCircularPolarizer(shape, z=0)(x_pol)
    right_field = RightCircularPolarizer(shape, z=0)(x_pol)


Waveplates
----------

Waveplates introduce a phase delay :math:`\phi` between fast and slow axes oriented at angle
:math:`\theta`:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Class
     - :math:`\phi`
     - Effect
   * - :class:`~torchoptics.elements.Waveplate`
     - Any
     - General waveplate with arbitrary phase delay.
   * - :class:`~torchoptics.elements.QuarterWaveplate`
     - :math:`\pi/2`
     - Linear ↔ circular conversion.
   * - :class:`~torchoptics.elements.HalfWaveplate`
     - :math:`\pi`
     - Rotates linear polarization by :math:`2\theta`.

.. code-block:: python

    from torchoptics.elements import QuarterWaveplate, HalfWaveplate

    circular = QuarterWaveplate(shape, theta=torch.pi/4, z=0)(x_pol)
    rotated = HalfWaveplate(shape, theta=torch.pi/4, z=0)(x_pol)


Custom Polarization Elements
-----------------------------

:class:`~torchoptics.elements.PolarizedModulator`,
:class:`~torchoptics.elements.PolarizedPhaseModulator`, and
:class:`~torchoptics.elements.PolarizedAmplitudeModulator` accept arbitrary ``(3, 3, H, W)``
profiles:

.. code-block:: python

    from torchoptics.elements import PolarizedPhaseModulator

    phase = torch.zeros(3, 3, shape, shape)
    phase[0, 0] = 0.0        # No phase shift for x→x
    phase[1, 1] = torch.pi   # π phase shift for y→y
    custom_pol = PolarizedPhaseModulator(phase, z=0)


Polarizing Beam Splitter
-------------------------

:class:`~torchoptics.elements.PolarizingBeamSplitter` separates a field into its :math:`x` and
:math:`y` components:

.. code-block:: python

    from torchoptics.elements import PolarizingBeamSplitter

    field_x, field_y = PolarizingBeamSplitter(shape, z=0)(x_pol)


Propagation
-----------

Each polarization component propagates independently. All field propagation methods work
seamlessly with polarized data:

.. code-block:: python

    propagated = x_pol.propagate_to_z(0.5)
