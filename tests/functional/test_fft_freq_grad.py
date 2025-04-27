import pytest
import torch

from torchoptics.functional import fftfreq_grad


def test_fftfreq_grad_values():
    for n in [0, 1, 2, 3, 8, 16, 31, 32]:
        for d_value in [1e-9, 1e-3, 1e-1, 1e1, 1e7]:
            d = torch.tensor(d_value)
            expected = torch.fft.fftfreq(n, d=d_value)
            actual = fftfreq_grad(n, d)
            assert torch.allclose(actual, expected)


def test_fftfreq_grad_dtype_device():
    n = 16
    d_value = 0.17
    dtype = torch.double
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for requires_grad in [False, True]:
            d = torch.tensor(d_value, dtype=dtype, device=device, requires_grad=requires_grad)
            expected = torch.fft.fftfreq(n, d=d_value).to(dtype=dtype, device=device)
            actual = fftfreq_grad(n, d)
            assert actual.dtype == expected.dtype
            assert actual.device == expected.device
            assert actual.requires_grad == requires_grad
            assert torch.allclose(actual, expected)
