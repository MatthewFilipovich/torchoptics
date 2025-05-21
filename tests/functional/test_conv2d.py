import torch

from torchoptics.functional import conv2d_fft


def test_conv2d_fft() -> None:
    input = torch.randn(3, 1, 11, 23) + 1j * torch.randn(3, 1, 11, 23)
    weight = torch.randn(1, 1, 3, 5) + 1j * torch.randn(1, 1, 3, 5)
    conv2d_output = torch.nn.functional.conv2d(input, weight.flip(-1, -2))
    conv2d_fft_output = conv2d_fft(input, weight)
    assert torch.allclose(conv2d_output, conv2d_fft_output, atol=1e-5)
