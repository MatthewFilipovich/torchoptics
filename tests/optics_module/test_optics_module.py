import pytest
import torch
from torch import Tensor
from torch.nn import Parameter

from torchoptics import OpticsModule


def test_initialization():
    module = OpticsModule()
    assert isinstance(module, OpticsModule)


def test_register_property_as_parameter():
    module = OpticsModule()
    value = torch.tensor([1.0, 2.0, 3.0])
    module.register_optics_property("prop1", Parameter(value))
    assert hasattr(module, "prop1")
    assert isinstance(module.prop1, Tensor)
    assert module.prop1.requires_grad
    assert "prop1" in dict(module.named_parameters())
    assert torch.equal(module.prop1, value)
    assert module.prop1 is not value


def test_register_property_as_buffer():
    module = OpticsModule()
    value = torch.tensor([1.0, 2.0, 3.0])
    module.register_optics_property("prop1", value)
    assert hasattr(module, "prop1")
    assert isinstance(module.prop1, Tensor)
    assert not module.prop1.requires_grad
    assert "prop1" in dict(module.named_buffers())


def test_set_property():
    module = OpticsModule()
    initial_value = torch.tensor([1.0, 2.0, 3.0])
    module.register_optics_property("prop1", initial_value)
    new_value = torch.tensor([4.0, 5.0, 6.0])
    module.set_optics_property("prop1", new_value)
    assert torch.equal(module.prop1, new_value)


def test_property_shape_validation():
    module = OpticsModule()
    initial_value = torch.tensor([1.0, 2.0, 3.0])
    module.register_optics_property("prop1", initial_value)
    new_value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        module.set_optics_property("prop1", new_value)


def test_complex_property_registration():
    module = OpticsModule()
    value = torch.tensor([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
    module.register_optics_property("prop1", value, is_complex=True)
    assert module.prop1.is_complex()


def test_is_positive():
    module = OpticsModule()
    value = torch.tensor([1.0, 2.0, 3.0])
    module.register_optics_property("prop1", value, is_positive=True)
    invalid_value = torch.tensor([-1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        module.set_optics_property("prop1", invalid_value)


def test_register_property_before_init():
    module = OpticsModule()
    module._initialized = False
    with pytest.raises(AttributeError):
        module.register_optics_property("prop1", torch.tensor([1.0]))


def test_register_property_from_sequence():
    module = OpticsModule()
    value = [1.0, 2.0]
    module.register_optics_property("prop1", value, is_vector2=True)
    assert hasattr(module, "prop1")
    assert isinstance(module.prop1, Tensor)
    expected_tensor = torch.tensor([1.0, 2.0], dtype=torch.double)
    assert torch.equal(module.prop1, expected_tensor)
    assert not module.prop1.requires_grad
    assert "prop1" in dict(module.named_buffers())


def test_register_property_with_none_shape():
    module = OpticsModule()
    value = torch.tensor([1.0, 2.0, 3.0])
    module.register_optics_property("prop1", value)
    assert hasattr(module, "prop1")
    assert isinstance(module.prop1, Tensor)
    assert not module.prop1.requires_grad
    assert module.prop1.shape == (3,)


def test_register_trainable_property_with_none_shape():
    module = OpticsModule()
    value = torch.tensor([1.0, 2.0, 3.0])
    module.register_optics_property("prop1", Parameter(value))
    assert hasattr(module, "prop1")
    assert isinstance(module.prop1, Tensor)
    assert module.prop1.requires_grad
    assert module.prop1.shape == (3,)


def test_set_property_via_setattr():
    module = OpticsModule()
    initial_value = [1.0, 2.0]
    module.register_optics_property("prop1", initial_value, is_vector2=True)
    new_value = [4.0, 5.0]
    module.prop1 = new_value
    expected_tensor = torch.tensor(new_value, dtype=torch.double)
    assert torch.equal(module.prop1, expected_tensor)


def test_set_property_via_setattr_tensor():
    module = OpticsModule()
    initial_value = [1.0, 2.0, 3.0]
    module.register_optics_property("prop1", initial_value)
    new_value = torch.tensor([4.0, 5.0, 6.0])
    module.prop1 = new_value
    assert torch.equal(module.prop1, new_value)


def test_set_trainable_property_via_setattr():
    module = OpticsModule()
    initial_value = torch.tensor([1.0, 2.0, 3.0])
    module.register_optics_property("prop1", Parameter(initial_value))
    new_value = [4.0, 5.0, 6.0]
    with torch.no_grad():
        module.prop1 = new_value
    expected_tensor = torch.tensor(new_value, dtype=torch.double)
    assert torch.equal(module.prop1, expected_tensor)


def test_raise_errors():
    with pytest.raises(AttributeError):
        OpticsModule().set_optics_property("unregistered_prop", 1.0)
