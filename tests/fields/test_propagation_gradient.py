import pytest
import torch
from torch.nn import Parameter

from torchoptics import Field, System

PROPAGATION_METHODS = ["ASM", "DIM"]


@pytest.mark.parametrize("propagation_method", PROPAGATION_METHODS)
def test_propagation_gradient_through_z(propagation_method):
    """Gradient flows through field.z during propagation."""
    field_z = Parameter(torch.tensor(1.0))
    field = Field(torch.ones(10, 10), z=field_z, wavelength=1, spacing=0.1)
    output = field.propagate_to_z(10.0, propagation_method=propagation_method)

    assert output.data.grad_fn is not None

    loss = output.data.abs().sum()
    loss.backward()

    assert field.z.grad is not None
    assert torch.isfinite(field.z.grad)


@pytest.mark.parametrize("propagation_method", PROPAGATION_METHODS)
def test_propagation_gradient_through_spacing(propagation_method):
    """Gradient flows through field.spacing during propagation."""
    spacing = Parameter(torch.tensor([0.1, 0.1]))
    field = Field(torch.ones(10, 10), z=0, wavelength=1, spacing=spacing)
    output = field.propagate_to_z(10.0, propagation_method=propagation_method)

    loss = output.data.abs().sum()
    loss.backward()

    assert spacing.grad is not None
    assert torch.all(torch.isfinite(spacing.grad))


@pytest.mark.parametrize("propagation_method", PROPAGATION_METHODS)
def test_propagation_gradient_through_offset(propagation_method):
    """Gradient flows through field.offset during propagation."""
    offset = Parameter(torch.tensor([0.0, 0.0]))
    field = Field(torch.ones(10, 10), z=0, wavelength=1, spacing=0.1, offset=offset)
    output = field.propagate_to_z(10.0, propagation_method=propagation_method)

    loss = output.data.abs().sum()
    loss.backward()

    assert field.offset.grad is not None
    assert torch.all(torch.isfinite(field.offset.grad))


@pytest.mark.parametrize("propagation_method", PROPAGATION_METHODS)
def test_propagation_gradient_through_data(propagation_method):
    """Gradient flows through field.data during propagation."""
    data = Parameter(torch.ones(10, 10, dtype=torch.cfloat))
    field = Field(data, z=0, wavelength=1, spacing=0.1)
    output = field.propagate_to_z(10.0, propagation_method=propagation_method)

    loss = output.data.abs().sum()
    loss.backward()

    assert field.data.grad is not None
    assert torch.all(torch.isfinite(field.data.grad.real))


def test_system_propagation_gradient_through_z():
    """Gradient flows through field.z when using System.measure_at_z."""
    field_z = Parameter(torch.tensor(0.0))
    field = Field(torch.ones(10, 10), z=field_z, wavelength=1, spacing=0.1)
    system = System()
    output = system.measure_at_z(field, z=10.0)

    loss = output.data.abs().sum()
    loss.backward()

    assert field.z.grad is not None
    assert torch.isfinite(field.z.grad)


@pytest.mark.parametrize("propagation_method", PROPAGATION_METHODS)
def test_propagation_gradient_output_field_connected(propagation_method):
    """Output field data is connected to the computation graph."""
    field_z = Parameter(torch.tensor(1.0))
    field = Field(torch.ones(10, 10), z=field_z, wavelength=1, spacing=0.1)
    output = field.propagate_to_z(10.0, propagation_method=propagation_method)

    assert output.data.grad_fn is not None
    assert output.data.requires_grad
