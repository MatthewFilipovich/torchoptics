"""This module defines functions for field propagation."""

from .propagator import (
    VALID_INTERPOLATION_MODES,
    VALID_PROPAGATION_METHODS,
    calculate_critical_propagation_distance,
    get_propagation_plane,
    propagator,
    use_angular_spectrum_method,
)

__all__ = [
    "VALID_INTERPOLATION_MODES",
    "VALID_PROPAGATION_METHODS",
    "calculate_critical_propagation_distance",
    "get_propagation_plane",
    "use_angular_spectrum_method",
    "propagator",
]
