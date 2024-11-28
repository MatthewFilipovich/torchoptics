"""This module defines functions for field propagation."""

from .propagator import (
    calculate_min_dim_propagation_distance,
    get_propagation_plane,
    is_dim_propagation,
    propagator,
)

VALID_PROPAGATION_METHODS = {"AUTO", "AUTO_FRESNEL", "ASM", "ASM_FRESNEL", "DIM", "DIM_FRESNEL"}
VALID_INTERPOLATION_MODES = {"none", "bilinear", "bicubic", "nearest"}
