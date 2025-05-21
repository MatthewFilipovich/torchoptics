import pytest

from torchoptics.utils import initialize_shape


def test_initialize_shape_valid() -> None:
    shape = initialize_shape([3, 4])
    assert shape == (3, 4)


def test_initialize_shape_invalid_vector() -> None:
    with pytest.raises(ValueError):
        initialize_shape([3, 4, 5])
