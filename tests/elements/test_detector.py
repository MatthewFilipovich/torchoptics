import pytest
import torch
from matplotlib.figure import Figure

from torchoptics import Field
from torchoptics.elements import LinearDetector


def test_linear_detector():
    shape = (100, 100)
    spacing = 1
    field = Field(torch.ones(*shape), wavelength=700e-9, spacing=spacing)
    weight = torch.zeros(2, *shape)
    weight[0, :50, :60] = 1
    weight[1, :40, :30] = 1

    detector = LinearDetector(weight, spacing=spacing)
    output = detector(field)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2,)
    assert torch.allclose(output, torch.tensor([3000.0, 1200.0], dtype=torch.double))
    fig = detector.visualize(0, show=False, return_fig=True)
    assert isinstance(fig, Figure)

    with pytest.raises(TypeError):
        LinearDetector("not a tensor", spacing=spacing)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LinearDetector(torch.rand(1, 2, 3, 4), spacing=spacing)
