import pytest
import torch

from torchoptics.config import get_default_dtype, set_default_dtype
from torchoptics.utils import initialize_tensor


def test_initialize_tensor_errors():
    with pytest.raises(ValueError):
        initialize_tensor("name", 1.0, is_scalar=True, is_complex=True, is_integer=True)
    with pytest.raises(ValueError):
        initialize_tensor("name", 1.1, is_scalar=True, is_complex=False, is_integer=True)


def test_initialize_scalar_tensor():
    tensor = initialize_tensor("scalar", 1.0, is_scalar=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 1.0


def test_scalar_and_vector2():
    with pytest.raises(ValueError):
        initialize_tensor("scalar_and_vector2", 1.0, is_scalar=True, is_vector2=True)


def test_initialize_vector2_tensor():
    tensor = initialize_tensor("vector2", [1.0, 2.0], is_vector2=True)
    assert torch.is_tensor(tensor)
    assert tensor.tolist() == [1.0, 2.0]


def test_initialize_vector2_from_scalar_preserves_configured_dtype():
    original_dtype = get_default_dtype()
    try:
        set_default_dtype(torch.float64)
        tensor = initialize_tensor("vector2", 1.0, is_vector2=True)
        assert tensor.dtype == torch.float64
        assert tensor.tolist() == [1.0, 1.0]
    finally:
        set_default_dtype(original_dtype)


def test_initialize_vector2_from_scalar_tensor_preserves_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scalar = torch.tensor(1.0, device=device)
    tensor = initialize_tensor("vector2", scalar, is_vector2=True)
    assert tensor.device == scalar.device


def test_initialize_complex_tensor():
    tensor = initialize_tensor("complex", 1.0 + 2.0j, is_complex=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 1.0 + 2.0j


def test_initialize_tensor_uses_configured_default_dtype():
    original_dtype = get_default_dtype()
    try:
        set_default_dtype(torch.float32)
        tensor = initialize_tensor("scalar", 1.0, is_scalar=True)
        assert tensor.dtype == torch.float32
    finally:
        set_default_dtype(original_dtype)


def test_initialize_complex_tensor_uses_configured_default_dtype():
    original_dtype = get_default_dtype()
    try:
        set_default_dtype(torch.float32)
        tensor = initialize_tensor("complex", 1.0 + 2.0j, is_complex=True)
        assert tensor.dtype == torch.complex64
    finally:
        set_default_dtype(original_dtype)


def test_initialize_integer_tensor():
    tensor = initialize_tensor("integer", 1, is_integer=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 1


def test_initialize_positive_tensor():
    tensor = initialize_tensor("positive", 1.0, is_positive=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 1.0


def test_initialize_non_negative_tensor():
    tensor = initialize_tensor("non_negative", 0.0, is_non_negative=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 0.0


def test_initialize_tensor_invalid_scalar():
    with pytest.raises(ValueError):
        initialize_tensor("invalid_scalar", [1.0, 2.0], is_scalar=True)


def test_initialize_tensor_invalid_vector2():
    with pytest.raises(ValueError):
        initialize_tensor("invalid_vector2", [1.0, 2.0, 3.0], is_vector2=True)


def test_initialize_tensor_invalid_positive():
    with pytest.raises(ValueError):
        initialize_tensor("invalid_positive", -1.0, is_positive=True)


def test_initialize_tensor_invalid_non_negative():
    with pytest.raises(ValueError):
        initialize_tensor("invalid_non_negative", -1.0, is_non_negative=True)
