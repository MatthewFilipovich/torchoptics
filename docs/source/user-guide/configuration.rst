Configuration
=============


TorchOptics uses two global defaults — **spacing** and **wavelength** — that are applied
whenever a :class:`~torchoptics.Field` or :class:`~torchoptics.elements.Element` is created
without explicitly specifying them. Set these at the start of every script.


Setting Defaults
----------------

.. code-block:: python

    import torchoptics

    torchoptics.set_default_spacing(10e-6)       # 10 µm grid spacing
    torchoptics.set_default_wavelength(700e-9)    # 700 nm wavelength

Retrieve the current values with the corresponding getters:

.. code-block:: python

    torchoptics.get_default_spacing()       # Returns Tensor
    torchoptics.get_default_wavelength()    # Returns Tensor

If you create a field or element without setting defaults first (and without passing ``spacing``
or ``wavelength`` explicitly), a ``ValueError`` is raised.


Spacing
-------

The spacing :math:`\Delta` is the physical distance between adjacent grid points. It controls
both the physical extent and the frequency bandwidth of the simulation:

.. math::

    L = N \cdot \Delta
    \qquad\qquad
    f_\text{max} = \frac{1}{2\Delta}

where :math:`N` is the number of grid points. Smaller spacing resolves higher spatial
frequencies but reduces the physical field of view for a given grid size.

Spacing can be **isotropic** (scalar) or **anisotropic** (2-element tuple):

.. code-block:: python

    torchoptics.set_default_spacing(10e-6)            # 10 µm × 10 µm
    torchoptics.set_default_spacing((10e-6, 20e-6))   # 10 µm × 20 µm


Wavelength
----------

The wavelength :math:`\lambda` sets the monochromatic operating wavelength used by propagation
methods, wavelength-dependent elements like :class:`~torchoptics.elements.Lens`, and profile
functions like :func:`~torchoptics.profiles.gaussian`.


Per-Object Overrides
--------------------

Every constructor accepts explicit ``spacing`` and ``wavelength`` values that override the
global defaults for that object only:

.. code-block:: python

    from torchoptics import Field, PlanarGrid
    from torchoptics.elements import Lens

    field = Field(data, wavelength=532e-9, spacing=5e-6)
    lens = Lens(300, focal_length=100e-3, spacing=5e-6)
    grid = PlanarGrid(shape=300, z=0, spacing=5e-6)


Data Type
---------

All tensors default to ``torch.float64`` (double precision), which provides high numerical
accuracy for phase-sensitive wave optics. Switch to single precision for faster computation:

.. code-block:: python

    torchoptics.set_default_dtype(torch.float32)

Only ``torch.float32`` and ``torch.float64`` are supported.


GPU and Device
--------------

All TorchOptics objects are PyTorch modules; move them to the GPU with ``.to(device)``
for significant speedups on large grids and optimization loops:

.. code-block:: python

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    field = field.to(device)
    system = system.to(device)
