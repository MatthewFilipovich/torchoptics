import torch

from torchoptics.functional import conv2d_fft


def test_conv2d_fft():
    input = torch.randn(3, 1, 11, 23) + 1j * torch.randn(3, 1, 11, 23)
    weight = torch.randn(1, 1, 3, 5) + 1j * torch.randn(1, 1, 3, 5)
    conv2d_output = torch.nn.functional.conv2d(input, weight.flip(-1, -2))
    conv2d_fft_output = conv2d_fft(input, weight)
    assert torch.allclose(conv2d_output, conv2d_fft_output, atol=1e-5)


def test_conv2d_fft_large_kernel():
    input = torch.randn(1, 1, 64, 96, dtype=torch.complex128)
    weight = torch.randn(1, 1, 17, 33, dtype=torch.complex128)
    conv2d_output = torch.nn.functional.conv2d(input, weight.flip(-1, -2))
    conv2d_fft_output = conv2d_fft(input, weight)
    assert torch.allclose(conv2d_output, conv2d_fft_output, atol=1e-8)


def test_conv2d_fft_with_padding_matches_conv2d():
    # Ensure fft_padding does not change numerical result compared to conv2d
    input = torch.randn(2, 1, 30, 45, dtype=torch.complex64)
    weight = torch.randn(1, 1, 5, 7, dtype=torch.complex64)
    expected = torch.nn.functional.conv2d(input, weight.flip(-1, -2))

    # no padding
    out0 = conv2d_fft(input, weight, fft_padding=0)
    # small padding
    out8 = conv2d_fft(input, weight, fft_padding=8)
    # larger padding
    out32 = conv2d_fft(input, weight, fft_padding=32)

    assert torch.allclose(expected, out0, atol=1e-5)
    assert torch.allclose(expected, out8, atol=1e-5)
    assert torch.allclose(expected, out32, atol=1e-5)
