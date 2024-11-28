import unittest

from torchoptics.param import Param


class TestParam(unittest.TestCase):
    def test_initialization(self):
        data = 42
        param = Param(data)
        self.assertEqual(param.data, data)

    def test_getattr(self):
        class ExampleClass:
            def __init__(self):
                self.value = 100

        dummy = ExampleClass()
        param = Param(dummy)
        self.assertEqual(param.value, dummy.value)


if __name__ == "__main__":
    unittest.main()
