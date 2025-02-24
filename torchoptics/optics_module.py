"""This module defines the OpticsModule class."""

from typing import Any

from torch.nn import Module, Parameter

from .utils import initialize_tensor

__all__ = ["OpticsModule"]


class OpticsModule(Module):  # pylint: disable=abstract-method
    """
    Base class for all optics modules.

    This class facilitates the registration of tensors, representing optics-related properties, as either
    PyTorch parameters or buffers. These properties are validated and registered using
    :meth:`register_optics_property()`::

        from torchoptics import OpticsModule
        from torch.nn import Parameter

        class MyOpticsModule(OpticsModule):
            def __init__(self, trainable_property, non_trainable_property):
                super().__init__()
                self.register_optics_property("trainable_property", Parameter(trainable_property), shape=())
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

    def register_optics_property(self, name: str, value: Any, **kwargs) -> None:
        """
        Registers an optics property as a PyTorch parameter or buffer.

        Args:
            name (str): Name of the optics property.
            value (Any): Initial value of the property.
            is_scalar (bool): Whether the property tensor is a scalar. Default: `False`.
            is_vector2 (bool): Whether the property tensor is a 2D vector. Default: `False`.
            is_complex (bool): Whether the property tensor is complex. Default: `False`.
            is_positive (bool): Whether to validate that the property tensor contains only positive
                values. Default: `False`.
            is_non_negative (bool): Whether to validate that the property tensor contains only
                non-negative. Default: `False`.
        """
        if not self._initialized:
            raise AttributeError("Cannot register optics property before __init__() call.")
        tensor = initialize_tensor(name, value, **kwargs)
        self._optics_property_configs[name] = kwargs

        if isinstance(value, Parameter):
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
            updated_tensor = initialize_tensor(name, value, **self._optics_property_configs[name])
            attr_tensor = getattr(self, name)
            if updated_tensor.shape != attr_tensor.shape:
                raise ValueError(
                    f"Cannot set {name} with shape {updated_tensor.shape}. "
                    f"Expected shape: {attr_tensor.shape}."
                )
            attr_tensor.copy_(updated_tensor)
        else:
            raise AttributeError(f"Cannot set unknown optics property: {name}.")
