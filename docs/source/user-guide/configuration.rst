.. _user-guide-configuration:

Configuration
==============

TorchOptics provides global configuration settings that simplify simulations by establishing
default values for commonly used parameters.

.. contents:: On This Page
   :local:
   :depth: 2

Global Defaults
----------------

Two parameters are used throughout TorchOptics and can be configured globally to avoid repetitive
specification:

- **Spacing**: The physical distance between adjacent grid points (in meters)
- **Wavelength**: The wavelength of monochromatic light (in meters)

Setting Defaults
^^^^^^^^^^^^^^^^^

Use :func:`~torchoptics.set_default_spacing` and :func:`~torchoptics.set_default_wavelength` to
establish global defaults at the start of your simulation:

.. code-block:: python

    import torchoptics

    torchoptics.set_default_spacing(10e-6)      # 10 μm grid spacing
    torchoptics.set_default_wavelength(700e-9)   # 700 nm wavelength

Once set, these values are used automatically whenever ``spacing`` or ``wavelength`` is not
explicitly provided:

.. code-block:: python

    from torchoptics import Field
    from torchoptics.profiles import gaussian

    # Both use the default spacing and wavelength
    profile = gaussian(200, waist_radius=500e-6)
    field = Field(profile)

Getting Defaults
^^^^^^^^^^^^^^^^^

Retrieve the current default values with:

.. code-block:: python

    spacing = torchoptics.get_default_spacing()
    wavelength = torchoptics.get_default_wavelength()

Overriding Defaults
^^^^^^^^^^^^^^^^^^^^

Any function or class that accepts ``spacing`` or ``wavelength`` can override the global default
by providing an explicit value:

.. code-block:: python

    # Uses default spacing
    field1 = Field(data)

    # Overrides with custom spacing
    field2 = Field(data, spacing=20e-6)

    # Profile with custom spacing
    profile = gaussian(200, waist_radius=500e-6, spacing=5e-6)

This allows you to mix different resolutions or wavelengths within the same simulation when needed.

Anisotropic Spacing
^^^^^^^^^^^^^^^^^^^^

Spacing can be specified as a single value (isotropic) or a tuple ``(dy, dx)`` for anisotropic grids:

.. code-block:: python

    # Isotropic: same spacing in both dimensions
    torchoptics.set_default_spacing(10e-6)

    # Anisotropic: different spacing in y and x
    torchoptics.set_default_spacing((10e-6, 20e-6))


When to Set Defaults
---------------------

**Best practice**: Set defaults once at the beginning of your script:

.. code-block:: python

    import torchoptics

    # Set up the simulation environment
    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    # ... rest of your simulation ...

**Multiple wavelengths**: If your simulation uses multiple wavelengths (e.g., polychromatic light),
set the default to the most commonly used wavelength and override where needed:

.. code-block:: python

    torchoptics.set_default_wavelength(550e-9)  # Green (most common)

    # Red channel
    red_field = Field(data, wavelength=700e-9)

    # Green channel (uses default)
    green_field = Field(data)

    # Blue channel
    blue_field = Field(data, wavelength=450e-9)


Floating-Point Precision
-------------------------

TorchOptics uses single precision (``float32``) by default, following PyTorch's convention. For
simulations that require higher numerical accuracy — especially when using Rayleigh-Sommerfeld
propagation methods (``ASM_RS``, ``DIM_RS``, ``AUTO_RS``) — you can switch to double precision:

.. code-block:: python

    import torch
    torch.set_default_dtype(torch.float64)

This should be set **before** creating any fields or optical elements. See
:ref:`user-guide-precision` for more details on when double precision is recommended.
