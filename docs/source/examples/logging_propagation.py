"""
Logging Propagation
===================

Displays log messages during field propagation.
"""

# %%
# Set up logging to display propagation events
import logging

logging.basicConfig(level=logging.INFO)


# %%
# Create an example field with a circular profile
import torchoptics
from torchoptics import Field
from torchoptics.profiles import circle

# Set default simulation parameters
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

# Initialize field with a circular profile
field = Field(circle(100, radius=50e-5))

# %%
# Propagate to z = 0 with a different shape; this triggers interpolation but no physical propagation
output_field = field.propagate(shape=200, z=0)

# %%
# Propagate to z = 0.01 m using the Angular Spectrum Method (ASM), which includes interpolation
output_field = field.propagate_to_z(0.01, propagation_method="ASM")

# %%
# Propagate to z = 0.03 m using the Direct Integration Method (DIM)
output_field = field.propagate_to_z(0.03, propagation_method="DIM")

# %%
# Propagate to z = 0.01 m using 'auto' method, which selects the optimal method based on distance
output_field = field.propagate_to_z(0.01, propagation_method="auto")
# %%
# Propagate to z = 0.02 m using 'auto' method
output_field = field.propagate_to_z(0.02, propagation_method="auto")
# %%
# Propagate to z = 0.03 m using 'auto' method
output_field = field.propagate_to_z(0.03, propagation_method="auto")
