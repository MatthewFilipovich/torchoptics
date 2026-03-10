.. raw:: html

   <style>
   .bd-sidebar-secondary {
       display: none;
   }
   </style>

.. toctree::
   :hidden:

   quickstart/index
   user-guide/index
   examples/index
   api-reference/index

TorchOptics Documentation
==========================

**TorchOptics** is a Python library for differentiable wave optics simulations built on
`PyTorch <https://pytorch.org/>`_. It provides a complete framework for modeling, analyzing, and
optimizing optical systems using Fourier optics, with full support for GPU acceleration, automatic
differentiation, and batch processing.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 🌊 Differentiable Wave Optics
      :class-card: sd-bg-light sd-border sd-shadow

      Model, analyze, and optimize optical systems using Fourier optics with both
      the Angular Spectrum Method and Direct Integration Method.

   .. grid-item-card:: 🔥 Built on PyTorch
      :class-card: sd-bg-light sd-border sd-shadow

      GPU acceleration, batch processing, and automatic differentiation — train optical
      hardware parameters with gradient-based optimization.

   .. grid-item-card:: 🛠️ End-to-End Optimization
      :class-card: sd-bg-light sd-border sd-shadow

      Jointly optimize optical hardware and machine learning models by backpropagating
      through the full physical simulation.

   .. grid-item-card:: 🔬 Optical Elements
      :class-card: sd-bg-light sd-border sd-shadow

      Lenses, modulators, detectors, beam splitters, polarizers, waveplates, and more
      — all differentiable and composable.

   .. grid-item-card:: 🖼️ Spatial Profiles
      :class-card: sd-bg-light sd-border sd-shadow

      Hermite-Gaussian, Laguerre-Gaussian, Bessel beams, Zernike polynomials, gratings,
      and geometric shapes.

   .. grid-item-card:: 🔆 Polarization & Coherence
      :class-card: sd-bg-light sd-border sd-shadow

      Full vectorial field support with Jones calculus and spatially incoherent light
      via Gaussian-Schell model coherence.


Quick Examples
--------------

**Simulate an optical system** — Image a Siemens star through a 4f relay:

.. code-block:: python

    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import Lens
    from torchoptics.profiles import siemens_star

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    shape = (1000, 1000)
    input_field = Field(siemens_star(shape, num_spokes=36, radius=4e-3))

    f = 200e-3
    system = System(
        Lens(shape, f, z=1 * f),
        Lens(shape, f, z=3 * f),
    )

    output = system.measure_at_z(input_field, z=4 * f)
    output.visualize(title="4f System Output")

**Optimize an optical element** — Learn phase masks that convert a Gaussian beam into a
Laguerre-Gaussian donut mode:

.. code-block:: python

    import torch
    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import PhaseModulator
    from torchoptics.profiles import gaussian, laguerre_gaussian

    torchoptics.set_default_spacing(10e-6)
    torchoptics.set_default_wavelength(700e-9)

    shape = (250, 250)
    input_field = Field(gaussian(shape, waist_radius=500e-6))
    target = Field(laguerre_gaussian(shape, p=0, l=1, waist_radius=500e-6), z=0.6)

    system = System(
        PhaseModulator(torch.nn.Parameter(torch.zeros(shape)), z=0.0),
        PhaseModulator(torch.nn.Parameter(torch.zeros(shape)), z=0.2),
        PhaseModulator(torch.nn.Parameter(torch.zeros(shape)), z=0.4),
    )

    optimizer = torch.optim.Adam(system.parameters(), lr=0.1)
    for _ in range(200):
        optimizer.zero_grad()
        output = system.measure_at_z(input_field, z=0.6)
        loss = 1 - output.inner(target).abs().square()
        loss.backward()
        optimizer.step()


.. _installation:

Installation
------------

TorchOptics requires Python 3.10+ and is available on `PyPI <https://pypi.org/project/torchoptics>`_:

.. code-block:: bash

    pip install torchoptics

This will automatically install PyTorch and all other required dependencies.

To install the latest development version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/MatthewFilipovich/torchoptics.git


Getting Started
---------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 📖 Quickstart
      :link: quickstart/index
      :link-type: doc

      Walk through the fundamentals: creating fields, propagating light, using lenses,
      and building optical systems.

   .. grid-item-card:: 📚 User Guide
      :link: user-guide/index
      :link-type: doc

      In-depth coverage of optical fields, elements, propagation methods, profiles,
      polarization, spatial coherence, optimization, and more.

   .. grid-item-card:: 💡 Examples
      :link: examples/index
      :link-type: doc

      Practical examples including 4f systems, diffraction patterns, diffractive splitter
      training, and polychromatic simulations.

   .. grid-item-card:: 📘 API Reference
      :link: api-reference/index
      :link-type: doc

      Complete reference for all classes, functions, and modules in the library.


Contributing
--------------

We welcome contributions! See our `Contributing Guide <https://github.com/MatthewFilipovich/torchoptics/blob/main/CONTRIBUTING.md>`_ for details.

Citing TorchOptics
-------------------

If you use TorchOptics in your research, please cite our `paper <https://arxiv.org/abs/2411.18591>`_:

.. code-block:: bibtex

    @article{filipovich2024torchoptics,
      title={TorchOptics: An open-source Python library for differentiable Fourier optics simulations},
      author={Filipovich, Matthew J and Bhatt, Parth and Mohanty, Saswata and Bhattacharya, Manas},
      journal={arXiv preprint arXiv:2411.18591},
      year={2024}
    }

License
-------

TorchOptics is distributed under the MIT License. See the `LICENSE <https://github.com/MatthewFilipovich/torchoptics/blob/main/LICENSE>`_ file for more details.