import pytest
import torch

from torchoptics.profiles import zernike


def make_zernike_args():
    return dict(shape=(100, 100), radius=50.0, spacing=(1.0, 1.0), offset=(0.0, 0.0))


def test_zernike_profile():
    args = make_zernike_args()
    n = 3
    m = 1
    profile = zernike(n=n, m=m, **args)  # type: ignore
    assert profile.shape == args["shape"]
    assert not torch.is_complex(profile)
    assert torch.all(profile >= -1)
    assert torch.all(profile <= 1)
    assert profile.dtype == torch.double


def test_invalid_zernike_parameters():
    args = make_zernike_args()
    with pytest.raises(ValueError):
        zernike(n=2, m=3, **args)  # type: ignore
    with pytest.raises(ValueError):
        zernike(n=3, m=2, **args)  # type: ignore
