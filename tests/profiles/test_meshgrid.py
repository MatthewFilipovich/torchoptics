import torch

from torchoptics import Field, System
from torchoptics.elements import Modulator
from torchoptics.profiles import circle


def test_profile_meshgrid_offset() -> None:
    # Verify that applying the offset in the profile or in the PlanarGrid yields identical results
    shape = (500, 500)
    spacing = 1
    radius = 20
    offset = (100, -200)

    # Offset applied in the profile
    system0 = System(Modulator(circle(shape, radius, spacing, offset), spacing=spacing))
    # Offset applied in the PlanarGrid
    system1 = System(Modulator(circle(shape, radius, spacing), spacing=spacing, offset=offset))

    input_field = Field(torch.ones(shape), 1, 0, 1)
    output_field0 = system0.measure_at_z(input_field, 0)
    output_field1 = system1.measure_at_z(input_field, 0)

    assert torch.allclose(output_field0.data, output_field1.data, atol=1e-5), "Output fields do not match!"
