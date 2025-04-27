import pytest

from torchoptics.utils import copy


def test_copy_function_with_property():
    class SampleClass:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self._c = c

        @property
        def c(self):
            return self._c

    obj = SampleClass(1, 2, 3)
    copied_obj = copy(obj)
    assert copied_obj.a == 1
    assert copied_obj.b == 2
    assert copied_obj.c == 3


def test_copy_function_with_updates():
    class SampleClass:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self._c = c

        @property
        def c(self):
            return self._c

    obj = SampleClass(1, 2, 3)
    copied_obj = copy(obj, a=4, c=5)
    assert copied_obj.a == 4
    assert copied_obj.b == 2
    assert copied_obj.c == 5


def test_copy_function_missing_attributes():
    class IncompleteClass:
        def __init__(self, a, b):
            self.a = a
            self.b_ = b

    obj = IncompleteClass(1, 2)
    with pytest.raises(ValueError):
        copy(obj)
