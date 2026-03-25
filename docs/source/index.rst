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

**TorchOptics** is an open-source Python library for simulating optical systems using `Fourier optics <https://en.wikipedia.org/wiki/Fourier_optics>`_, built on `PyTorch <https://pytorch.org/>`_. It provides GPU-accelerated, fully differentiable wave optics simulations, enabling end-to-end optimization of optical hardware jointly with machine learning models.

Key Features
------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 🌊 **Differentiable Wave Optics**
      :class-card: sd-border sd-shadow-sm sd-card-hover

      Model, analyze, and optimize optical systems using Fourier optics.

   .. grid-item-card:: 🔥 **Built on PyTorch**
      :class-card: sd-border sd-shadow-sm sd-card-hover

      GPU acceleration, batch processing, and automatic differentiation.

   .. grid-item-card:: 🛠️ **End-to-End Optimization**
      :class-card: sd-border sd-shadow-sm sd-card-hover

      Joint optimization of optical hardware and machine learning models.

   .. grid-item-card:: 🔬 **Optical Elements**
      :class-card: sd-border sd-shadow-sm sd-card-hover

      Lenses, phase/amplitude modulators, detectors, polarizers, and more.

   .. grid-item-card:: 🖼️ **Spatial Profiles**
      :class-card: sd-border sd-shadow-sm sd-card-hover

      Hermite-Gaussian, Laguerre-Gaussian, Zernike modes, gratings, and others.

   .. grid-item-card:: 🔆 **Polarization & Coherence**
      :class-card: sd-border sd-shadow-sm sd-card-hover

      Simulate polarized light and fields with arbitrary spatial coherence.

.. _installation:

Installation
------------

.. code-block:: bash

    pip install torchoptics


Contributing
--------------

Contributions are welcome! See the `Contributing Guide <https://github.com/MatthewFilipovich/torchoptics/blob/main/CONTRIBUTING.md>`_ for details.

Citing TorchOptics
-------------------

If you use TorchOptics in your research, please cite our
`paper <https://arxiv.org/abs/2411.18591>`_.

License
-------

Distributed under the MIT License. See `LICENSE <https://github.com/MatthewFilipovich/torchoptics/blob/main/LICENSE>`_ for details.