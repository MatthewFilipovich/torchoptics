"""This module defines the OpticsModule class."""

from typing import Any, Optional, Sequence

from torch import Tensor
from torch.nn import Module, Parameter

from .functional import initialize_tensor
from .param import Param

__all__ = ["OpticsModule"]


class OpticsModule(Module):  # pylint: disable=abstract-method
    """
    Base class for all optics modules.

    This class facilitates the registration of tensors, representing optics-related properties, as either
    PyTorch parameters or buffers. These properties are validated and registered using
    :meth:`register_optics_property()`::

        from torchoptics import OpticsModule, Param

        class MyOpticsModule(OpticsModule):
            def __init__(self, trainable_property, non_trainable_property):
                super().__init__()
                self.register_optics_property("trainable_property", Param(trainable_property), shape=())
                self.register_optics_property("non_trainable_property", non_trainable_property, shape=())

    Once the properties are registered, they can be updated using :meth:`set_optics_property()`.

    .. note::
        :meth:`__setattr__()` is overridden to call :meth:`set_optics_property()` when setting the value of an
        optics property.

    """

    _initialized = False

    def __init__(self) -> None:
        super().__init__()
        self._optics_property_configs: dict[str, dict] = {}
        self._initialized = True

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of the module.

        If the attribute name is a registered optics property, the optics property value is set using the
        :meth:`set_optics_property()` method. Otherwise, the attribute is set normally.

        Args:
            name (str): The name of the attribute.
            value (Any): The value to set for the attribute.
        """
        if self._initialized and name in self._optics_property_configs:
            self.set_optics_property(name, value)
        else:
            super().__setattr__(name, value)

    def register_optics_property(
        self,
        name: str,
        value: Any,
        shape: Optional[Sequence] = None,
        is_complex: bool = False,
        validate_positive: bool = False,
        validate_non_negative: bool = False,
        fill_value: bool = False,
    ) -> None:
        """
        Registers an optics property as a PyTorch parameter or buffer.

        Args:
            name (str): Name of the optics property.
            value (Any): Initial value of the property.
            shape (Optional[Sequence]): Shape of the property tensor. Required if value is not a tensor.
                Default: `None`.
            is_complex (bool): Whether the property tensor is complex. Default: `False`.
            validate_positive (bool): Whether to validate that the property tensor contains only positive
                values. Default: `False`.
            validate_non_negative (bool): Whether to validate that the property tensor contains only
                non-negative. Default: `False`.
            fill_value (bool): Whether to fill the tensor with the initial value if it is a scalar.
                Default: `False`.

        Raises:
            AttributeError: If called before the class is initialized.
            ValueError: If shape is not provided and value is not a tensor, or if the value does not match the
                property's conditions.
        """
        if not self._initialized:
            raise AttributeError("Cannot register optics property before __init__() call.")
        is_param = isinstance(value, Param)
        value = value.data if is_param else value
        if shape is None:
            if not isinstance(value, Tensor):
                raise ValueError(f"shape must be provided if {name} is not a tensor.")
            shape = value.shape

        property_config = {
            "name": name,
            "shape": shape,
            "is_complex": is_complex,
            "validate_positive": validate_positive,
            "validate_non_negative": validate_non_negative,
            "fill_value": fill_value,
        }
        self._optics_property_configs[name] = property_config

        tensor = initialize_tensor(value=value, **property_config)  # type: ignore
        if is_param:
            self.register_parameter(name, Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

    def set_optics_property(self, name: str, value: Any) -> None:
        """
        Sets the value of an existing optics property.

        Args:
            name (str): Name of the optics property.
            value (Any): New value of the property.

        Raises:
            AttributeError: If the property is not registered.
            ValueError: If the value does not match the property's conditions.
        """
        if self._initialized and name in self._optics_property_configs:
            updated_tensor = initialize_tensor(value=value, **self._optics_property_configs[name])
            attr_tensor = getattr(self, name)
            attr_tensor.copy_(updated_tensor)
        else:
            raise AttributeError(f"Cannot set unknown optics property: {name}.")
