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

### Simulate an optical system

Image a Siemens star resolution target through a 4f relay:

```python
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
```

### Optimize an optical element

Learn phase masks that convert a Gaussian beam into a Laguerre-Gaussian donut mode:

```python
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
```

_For more examples and detailed usage, please refer to the [documentation](https://torchoptics.readthedocs.io/)._

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/MatthewFilipovich/torchoptics/blob/main/CONTRIBUTING.md) for details.

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

TorchOptics is distributed under the MIT License. See the [LICENSE](https://github.com/MatthewFilipovich/torchoptics/blob/main/LICENSE) file for more details.
