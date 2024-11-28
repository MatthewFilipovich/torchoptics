"""This module defines the Param class, a wrapper class in TorchOptics to represent trainable parameters."""

from typing import Any

__all__ = ["Param"]


class Param:
    """
    A wrapper class for encapsulating trainable parameters in TorchOptics.

    Example:
        Initialize the ``z`` property of a :class:`Lens` object as a trainable parameter::

            from torchoptics import Param
            from torchoptics.elements import Lens

            lens = Lens(shape=1000, focal_length=0.2, z=Param(0.1), wavelength=700e-9, spacing=10e-6)

    Args:
        data (Any): The wrapped parameter value.
    """

    def __init__(self, data: Any):
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)
