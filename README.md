<p align="center">
  <img src="https://raw.githubusercontent.com/MatthewFilipovich/torchoptics/main/docs/source/_static/torchoptics_logo.png" width="700px">
</p>

<div align="center">

[![build](https://github.com/MatthewFilipovich/torchoptics/actions/workflows/build.yml/badge.svg)](https://github.com/MatthewFilipovich/torchoptics/actions/workflows/build.yml)
[![Codecov](https://img.shields.io/codecov/c/github/matthewfilipovich/torchoptics?token=52MBM273IF)](https://codecov.io/gh/MatthewFilipovich/torchoptics)
[![Documentation Status](https://readthedocs.org/projects/torchoptics/badge/?version=latest)](https://torchoptics.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/torchoptics.svg)](https://pypi.org/project/torchoptics/)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/MatthewFilipovich/torchoptics?color=blue)](https://github.com/MatthewFilipovich/torchoptics/blob/main/LICENSE)

</div>

> TorchOptics is a differentiable wave optics simulation library built on PyTorch.

# Key Features

- 🌊 **Differentiable Wave Optics** — Model, analyze, and optimize optical systems using Fourier optics.
- 🔥 **Built on PyTorch** — GPU acceleration, batch processing, and automatic differentiation.
- 🛠️ **End-to-End Optimization** — Joint optimization of optical hardware and machine learning models.
- 🔬 **Optical Elements** — Lenses, modulators, detectors, polarizers, and more.
- 🖼️ **Spatial Profiles** — Hermite-Gaussian, Laguerre-Gaussian, Zernike modes, and others.
- 🔆 **Polarization and Coherence** — Simulate polarized light and fields with arbitrary spatial coherence.

Learn more about TorchOptics in our research paper on [arXiv](https://arxiv.org/abs/2411.18591).

# Installation

TorchOptics is available on [PyPI](https://pypi.org/project/torchoptics/) and can be installed with:

```bash
pip install torchoptics
```

## Documentation

Read the full documentation at [torchoptics.readthedocs.io](https://torchoptics.readthedocs.io/).

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MatthewFilipovich/torchoptics/blob/main/docs/source/_static/torchoptics_colab.ipynb)

This example shows how to simulate a 4f imaging system using TorchOptics, computing and visualizing the field at each focal plane along the optical axis:

```python
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

# Configure default properties
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
```

<p align="center">
  <img src="https://raw.githubusercontent.com/MatthewFilipovich/torchoptics/main/docs/source/_static/4f_simulation.png" width="700px">
  <br>
  <em>Intensity distributions at different focal planes in the 4f system.</em>
</p>

<p align="center">
  <img width="300px" src="https://raw.githubusercontent.com/MatthewFilipovich/torchoptics/main/docs/source/_static/4f_propagation.gif">
  <br>
  <em>Propagation of the intensity distribution.</em>
</p>

_For more examples and detailed usage, please refer to the [documentation](https://torchoptics.readthedocs.io/)._

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/MatthewFilipovich/torchoptics/blob/main/CONTRIBUTING.md) for details.

## Citing TorchOptics

If you use TorchOptics in your research, please cite our paper:

> M.J. Filipovich and A.I. Lvovsky, _TorchOptics: An open-source Python library for differentiable Fourier optics simulations_, arXiv preprint [arXiv:2411.18591](https://arxiv.org/abs/2411.18591) (2024).

## License

TorchOptics is distributed under the MIT License. See the [LICENSE](https://github.com/MatthewFilipovich/torchoptics/blob/main/LICENSE) file for more details.
