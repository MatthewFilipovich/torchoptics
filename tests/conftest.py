import pytest
import torch

import torchoptics


@pytest.fixture(autouse=True)
def set_float32_defaults_for_tests():
    original_torchoptics_dtype = torchoptics.get_default_dtype()
    original_torch_dtype = torch.get_default_dtype()

    torchoptics.set_default_dtype(torch.float32)
    torch.set_default_dtype(torch.float32)

    try:
        yield
    finally:
        torchoptics.set_default_dtype(original_torchoptics_dtype)
        torch.set_default_dtype(original_torch_dtype)
