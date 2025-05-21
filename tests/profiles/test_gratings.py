from typing import TypedDict

import torch

from torchoptics.profiles import gratings
from torchoptics.type_defs import Scalar


class GratingArgs(TypedDict):
    shape: tuple[int, int]
    spacing: tuple[float, float]
    offset: tuple[float, float]
    theta: Scalar


def make_grating_args() -> GratingArgs:
    return GratingArgs(shape=(100, 100), spacing=(0.1, 0.1), offset=(0.0, 0.0), theta=0.0)


def test_blazed_grating():
    args = make_grating_args()
    period = 10.0
    height = 2.0
    profile = gratings.blazed_grating(period=period, height=height, **args)
    assert profile.shape == args["shape"]
    assert not torch.is_complex(profile)
    assert profile.dtype == torch.double


def test_sinusoidal_grating():
    args = make_grating_args()
    period = 10.0
    height = 1.0
    profile = gratings.sinusoidal_grating(period=period, height=height, **args)
    assert profile.shape == args["shape"]
    assert not torch.is_complex(profile)
    assert torch.all(profile >= -height)
    assert torch.all(profile <= height)
    assert profile.dtype == torch.double


def test_binary_grating():
    args = make_grating_args()
    period = 10.0
    profile = gratings.binary_grating(period=period, **args)
    assert profile.shape == args["shape"]
    assert not torch.is_complex(profile)
    assert torch.all((profile == 0) | (profile == 1))
    assert profile.dtype == torch.double
