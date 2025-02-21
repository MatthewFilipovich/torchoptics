TorchOptics Documentation
-------------------------

.. toctree::
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   introduction
   getting_started/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   API <autoapi/torchoptics/index>

.. toctree::
   :maxdepth: 1
   :caption: Background
   :hidden:

   background

.. image:: _static/torchoptics_logo.png
   :width: 700px
   :align: center

*TorchOptics is an open-source Python library for differentiable Fourier optics simulations with PyTorch.*

.. note::
    **Documentation is currently being developed.** For detailed information about the library, please refer to the `arXiv paper <https://arxiv.org/abs/2411.18591>`_.

Key Features
============

- **Differentiable Fourier Optics Simulations**: A comprehensive framework for modeling, analyzing, and designing optical systems using differentiable Fourier optics.
- **Built on PyTorch**: Leverages PyTorch for GPU acceleration, batch processing, automatic differentiation, and efficient gradient-based optimization.
- **End-to-End Optimization**: Enables optimization of optical hardware and deep learning models within a unified, differentiable pipeline.
- **Wide Range of Optical Elements and Spatial Profiles**: Includes standard elements like lenses and modulators, along with commonly used spatial profiles such as Hermite-Gaussian and Laguerre-Gaussian beams.
- **Polarized Light Simulation**: Simulates polarized light interactions using matrix Fourier optics with Jones calculus.
- **Spatial Coherence Support**: Models optical fields with arbitrary spatial coherence through the mutual coherence function.

Our research paper, available on `arXiv <https://arxiv.org/abs/2411.18591>`_, introduces the TorchOptics library and provides a comprehensive review of its features and applications.

Installation
============

TorchOptics and its dependencies can be installed using `pip <https://pypi.org/project/torchoptics/>`_:

.. code-block:: bash

    pip install torchoptics

To install the library in development mode, first clone the GitHub repository and then use pip to install it in editable mode:

.. code-block:: bash

    git clone https://github.com/MatthewFilipovich/torchoptics
    pip install -e ./torchoptics

Usage
=====

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/MatthewFilipovich/torchoptics/blob/main/docs/source/tutorials/4f_system.ipynb
   :alt: Open in Colab

This example demonstrates simulating a 4f imaging system using TorchOptics. The field at each focal plane along the z-axis is computed and visualized:

.. code-block:: python

    import torch
    import torchoptics
    from torchoptics import Field, System
    from torchoptics.elements import Lens
    from torchoptics.profiles import checkerboard

    # Set simulation properties
    shape = 1000  # Number of grid points in each dimension
    spacing = 10e-6  # Spacing between grid points (m)
    wavelength = 700e-9  # Field wavelength (m)
    focal_length = 200e-3  # Lens focal length (m)
    tile_length = 400e-6  # Checkerboard tile length (m)
    num_tiles = 15  # Number of tiles in each dimension

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure torchoptics default properties
    torchoptics.set_default_spacing(spacing)
    torchoptics.set_default_wavelength(wavelength)

    # Initialize input field with checkerboard pattern
    field_data = checkerboard(shape, tile_length, num_tiles)
    input_field = Field(field_data).to(device)

    # Define 4f optical system with two lenses
    system = System(
        Lens(shape, focal_length, z=1 * focal_length),
        Lens(shape, focal_length, z=3 * focal_length),
    ).to(device)

    # Measure field at focal planes along the z-axis
    measurements = [
        system.measure_at_z(input_field, z=i * focal_length)
        for i in range(5)
    ]

    # Visualize the measured intensity distributions
    for i, measurement in enumerate(measurements):
        measurement.visualize(title=f"z={i}f", vmax=1)

.. figure:: _static/4f_simulation.png
   :width: 700px
   :align: center

   Intensity distributions at different focal planes in the 4f system.

.. figure:: _static/4f_propagation.gif
   :width: 300px
   :align: center
   
   Propagation of the intensity distribution.

Contributing
============

We welcome all bug reports and suggestions for future features and enhancements, which can be filed as GitHub issues. To contribute a feature:

1. Fork it at `https://github.com/MatthewFilipovich/torchoptics/fork <https://github.com/MatthewFilipovich/torchoptics/fork>`_.
2. Create your feature branch (:code:`git checkout -b feature/fooBar`).
3. Commit your changes (:code:`git commit -am 'Add some fooBar'`).
4. Push to the branch (:code:`git push origin feature/fooBar`).
5. Submit a Pull Request.

Citing TorchOptics
===================

If you are using TorchOptics for research purposes, we kindly request that you cite the following paper:

    M.J. Filipovich and A.I. Lvovsky, *TorchOptics: An open-source Python library for differentiable Fourier optics simulations*, arXiv preprint `arXiv:2411.18591 <https://arxiv.org/abs/2411.18591>`_ (2024).

License
=======

TorchOptics is distributed under the MIT License. See the `LICENSE <https://github.com/MatthewFilipovich/torchoptics/blob/main/LICENSE>`_ file for more details.