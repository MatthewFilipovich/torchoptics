import pytest
import torch

from torchoptics.utils import initialize_tensor


def test_initialize_tensor_errors() -> None:
    with pytest.raises(ValueError):
        initialize_tensor("name", 1.0, is_scalar=True, is_complex=True, is_integer=True)
    with pytest.raises(ValueError):
        initialize_tensor("name", 1.1, is_scalar=True, is_complex=False, is_integer=True)


def test_initialize_scalar_tensor() -> None:
    tensor = initialize_tensor("scalar", 1.0, is_scalar=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 1.0


def test_scalar_and_vector2() -> None:
    with pytest.raises(ValueError):
        initialize_tensor("scalar_and_vector2", 1.0, is_scalar=True, is_vector2=True)


def test_initialize_vector2_tensor() -> None:
    tensor = initialize_tensor("vector2", [1.0, 2.0], is_vector2=True)
    assert torch.is_tensor(tensor)
    assert tensor.tolist() == [1.0, 2.0]


def test_initialize_complex_tensor() -> None:
    tensor = initialize_tensor("complex", 1.0 + 2.0j, is_complex=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 1.0 + 2.0j


def test_initialize_integer_tensor() -> None:
    tensor = initialize_tensor("integer", 1, is_integer=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 1


def test_initialize_positive_tensor() -> None:
    tensor = initialize_tensor("positive", 1.0, is_positive=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 1.0


def test_initialize_non_negative_tensor() -> None:
    tensor = initialize_tensor("non_negative", 0.0, is_non_negative=True)
    assert torch.is_tensor(tensor)
    assert tensor.item() == 0.0


def test_initialize_tensor_invalid_scalar() -> None:
    with pytest.raises(ValueError):
        initialize_tensor("invalid_scalar", [1.0, 2.0], is_scalar=True)


def test_initialize_tensor_invalid_vector2() -> None:
    with pytest.raises(ValueError):
        initialize_tensor("invalid_vector2", [1.0, 2.0, 3.0], is_vector2=True)


def test_initialize_tensor_invalid_positive() -> None:
    with pytest.raises(ValueError):
        initialize_tensor("invalid_positive", -1.0, is_positive=True)


def test_initialize_tensor_invalid_non_negative() -> None:
    with pytest.raises(ValueError):
        initialize_tensor("invalid_non_negative", -1.0, is_non_negative=True)
