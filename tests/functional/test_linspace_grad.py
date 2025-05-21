import torch

from torchoptics.functional import linspace_grad


def test_output_values() -> None:
    start_val, end_val = 0.0, 1.0
    steps = 10
    start = torch.tensor(start_val, requires_grad=True, dtype=torch.float64)
    end = torch.tensor(end_val, requires_grad=True, dtype=torch.float64)
    expected = torch.linspace(start_val, end_val, steps, dtype=torch.float64)
    actual = linspace_grad(start, end, steps)
    assert torch.allclose(actual, expected)


def test_edge_cases() -> None:
    for start_val, end_val, steps in [(1, 1, 0), (1, 3, 0), (12, 12, 1), (12, 14, 1)]:
        start = torch.tensor(start_val, requires_grad=True, dtype=torch.float64)
        end = torch.tensor(end_val, requires_grad=True, dtype=torch.float64)
        expected = torch.linspace(start_val, end_val, steps, dtype=torch.float64)
        actual = linspace_grad(start, end, steps)
        assert torch.allclose(actual, expected)


def test_differentiability() -> None:
    start = torch.tensor(-3, requires_grad=True, dtype=torch.float64)
    end = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    steps = 10
    linspace_tensor = linspace_grad(start, end, steps)
    output = linspace_tensor.sum()
    output.backward()
    assert start.grad is not None
    assert end.grad is not None


def test_dtype() -> None:
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for dtype in [torch.float32, torch.float64]:
            start = torch.tensor(0.0, requires_grad=True, device=device, dtype=dtype)
            end = torch.tensor(1.0, requires_grad=True, device=device, dtype=dtype)
            steps = 10
            linspace_tensor = linspace_grad(start, end, steps)
            assert linspace_tensor.device == start.device
            assert linspace_tensor.dtype == dtype
