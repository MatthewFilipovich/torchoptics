<p align="center">
  <img src="https://raw.githubusercontent.com/MatthewFilipovich/torchoptics/main/docs/source/_static/torchoptics_logo.png" width="700px">
</p>

<div align="center">

[![build](https://github.com/MatthewFilipovich/torchoptics/actions/workflows/build.yml/badge.svg)](https://github.com/MatthewFilipovich/torchoptics/actions/workflows/build.yml)
[![Codecov](https://img.shields.io/codecov/c/github/matthewfilipovich/torchoptics?token=52MBM273IF)](https://codecov.io/gh/MatthewFilipovich/torchoptics)
[![Documentation Status](https://readthedocs.org/projects/torchoptics/badge/?version=latest)](https://torchoptics.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/torchoptics.svg)](https://pypi.org/project/torchoptics/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/MatthewFilipovich/torchoptics?color=blue)](https://github.com/MatthewFilipovich/torchoptics/blob/main/LICENSE)

</div>

**TorchOptics** is an open-source Python library for simulating optical systems using [Fourier optics](https://en.wikipedia.org/wiki/Fourier_optics), built on [PyTorch](https://pytorch.org/). It provides GPU-accelerated, fully differentiable wave optics simulations, enabling end-to-end optimization of optical hardware jointly with machine learning models.

Learn more in our [paper on arXiv](https://arxiv.org/abs/2411.18591).

## Key Features

- 🌊 **Differentiable Wave Optics:** Model, analyze, and optimize optical systems using Fourier optics.
- 🔥 **Built on PyTorch:** GPU acceleration, batch processing, and automatic differentiation.
- 🛠️ **End-to-End Optimization:** Joint optimization of optical hardware and machine learning models.
- 🔬 **Optical Elements:** Lenses, phase/amplitude modulators, detectors, polarizers, and more.
- 🖼️ **Spatial Profiles:** Hermite-Gaussian, Laguerre-Gaussian, Zernike modes, gratings, and others.
- 🔆 **Polarization and Coherence:** Simulate polarized light and fields with arbitrary spatial coherence.


## Installation

```bash
pip install torchoptics
```

## Documentation

Full documentation is available at [torchoptics.readthedocs.io](https://torchoptics.readthedocs.io/).

## Examples

### Wave Propagation

Simulate free-space propagation of an octagonal aperture ([full example](https://torchoptics.readthedocs.io/en/stable/examples/optical_phenomena/animate_propagation.html)):

```python
import torch
import torchoptics
from torchoptics import Field
from torchoptics.profiles import octagon

device = "cuda" if torch.cuda.is_available() else "cpu"
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

field = Field(octagon(shape=500, radius=150e-5)).to(device)

for z in torch.linspace(0, 2, 11):
    field.propagate_to_z(z).visualize(title=f"z = {z:.2f} m")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/MatthewFilipovich/torchoptics/main/docs/source/_static/propagation_octagon.gif" width="200">
</p>

### 4f Imaging System

Simulate a 4f system with a high-pass spatial filter ([full example](https://torchoptics.readthedocs.io/en/stable/examples/optical_systems/4f_system.html)):
```python
import torch
import torchoptics
from torchoptics import Field, System
from torchoptics.elements import AmplitudeModulator, Lens
from torchoptics.profiles import checkerboard, circle

device = "cuda" if torch.cuda.is_available() else "cpu"
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

shape = 1000
f = 200e-3

input_field = Field(checkerboard(shape, tile_length=400e-6, num_tiles=15)).to(device)

system = System(
    Lens(shape, f, z=1 * f),
    AmplitudeModulator(1 - circle(shape, radius=5e-4), z=2 * f),
    Lens(shape, f, z=3 * f),
).to(device)

for i in range(5):
    system.measure_at_z(input_field, z=i * f).visualize(title=f"z={i}f", vmax=1)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/MatthewFilipovich/torchoptics/main/docs/source/_static/4f_system.png" width="700px">
</p>

### Inverse Design

Train a diffractive optical system to convert a Gaussian beam into a petal beam ([full example](https://torchoptics.readthedocs.io/en/stable/examples/optimization/training_petal_beam.html)):

```python
import torch
import torchoptics
from torch.nn import Parameter
from torchoptics import Field, System
from torchoptics.elements import PhaseModulator
from torchoptics.profiles import gaussian, laguerre_gaussian

device = "cuda" if torch.cuda.is_available() else "cpu"
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

shape = 250
waist_radius = 300e-6

input_field = Field(gaussian(shape, waist_radius=waist_radius), z=0).to(device)

petal_profile = laguerre_gaussian(shape, p=0, l=4, waist_radius=waist_radius)
petal_profile += laguerre_gaussian(shape, p=0, l=-4, waist_radius=waist_radius)
target_field = Field(petal_profile, z=0.8).normalize().to(device)

system = System(
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.2),
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.4),
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.6),
).to(device)

optimizer = torch.optim.Adam(system.parameters(), lr=0.05)
for iteration in range(100):
    optimizer.zero_grad()
    output_field = system.measure_at_z(input_field, 0.8)
    loss = 1 - output_field.inner(target_field).abs().square()
    loss.backward()
    optimizer.step()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/MatthewFilipovich/torchoptics/main/docs/source/_static/training_petal_beam.gif" width="700">
</p>

For more examples, see the [examples gallery](https://torchoptics.readthedocs.io/en/stable/examples/index.html).

## Contributing

Contributions are welcome! See the [Contributing Guide](https://github.com/MatthewFilipovich/torchoptics/blob/main/CONTRIBUTING.md) for details.

## Citing TorchOptics

If you use TorchOptics in your research, please cite our [paper](https://arxiv.org/abs/2411.18591):

```bibtex
@misc{filipovich2024torchoptics,
      title={TorchOptics: An open-source Python library for differentiable Fourier optics simulations},
      author={Matthew J. Filipovich and A. I. Lvovsky},
      year={2024},
      eprint={2411.18591},
      archivePrefix={arXiv},
      primaryClass={physics.optics},
      url={https://arxiv.org/abs/2411.18591},
}
```

## License

Distributed under the MIT License. See [LICENSE](https://github.com/MatthewFilipovich/torchoptics/blob/main/LICENSE) for details.
