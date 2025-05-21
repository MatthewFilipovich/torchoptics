import pytest
import torch

from torchoptics.profiles import special


def make_special_args():
    return {"shape": (100, 100), "spacing": (0.1, 0.1), "offset": (0.0, 0.0)}


def test_airy() -> None:
    args = make_special_args()
    scale = 10.0
    profile = special.airy(scale=scale, **args)
    assert profile.shape == args["shape"]
    assert torch.all(profile >= 0)
    assert profile.dtype == torch.double


def test_sinc() -> None:
    args = make_special_args()
    scale = (10.0, 20.0)
    profile = special.sinc(scale=scale, **args)
    assert profile.shape == args["shape"]
    assert torch.all(profile >= 0)
    assert profile.dtype == torch.double


def test_siemens_star() -> None:
    args = make_special_args()
    num_spokes = 8
    radius = 20.0
    profile = special.siemens_star(num_spokes=num_spokes, radius=radius, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
    assert profile.dtype == torch.double
    with pytest.raises(ValueError):
        special.siemens_star(num_spokes=num_spokes + 1, radius=radius, **args)
