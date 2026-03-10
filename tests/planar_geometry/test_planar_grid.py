from unittest.mock import patch

import torch
from matplotlib.figure import Figure

import torchoptics
from torchoptics import PlanarGrid


def make_planar_grid():
    shape = (100, 101)
    z = 1.0
    spacing = (0.1, 0.2)
    offset = (0.0, 0.0)
    return shape, z, spacing, offset


def test_initialization():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    assert plane.shape == shape
    assert torch.equal(plane.z, torch.tensor(z, dtype=torch.double))
    assert torch.equal(plane.spacing, torch.tensor(spacing, dtype=torch.double))
    assert torch.equal(plane.offset, torch.tensor(offset, dtype=torch.double))


def test_shape():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    assert isinstance(plane.shape, tuple)
    assert len(plane.shape) == 2
    assert all(isinstance(s, int) for s in plane.shape)


def test_default_initialization():
    torchoptics.set_default_spacing((0.1, 0.1))
    shape, z, _, _ = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z)
    assert torch.equal(plane.spacing, torch.tensor((0.1, 0.1), dtype=torch.double))


def test_geometry_property():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    expected_geometry = {
        "shape": shape,
        "z": torch.tensor(z, dtype=torch.double),
        "spacing": torch.tensor(spacing, dtype=torch.double),
        "offset": torch.tensor(offset, dtype=torch.double),
    }
    assert plane.geometry["shape"] == expected_geometry["shape"]
    assert torch.equal(plane.geometry["z"], expected_geometry["z"])
    assert torch.equal(plane.geometry["spacing"], expected_geometry["spacing"])
    assert torch.equal(plane.geometry["offset"], expected_geometry["offset"])


def test_grid_cell_area():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    expected_area = torch.tensor(spacing[0] * spacing[1], dtype=torch.double)
    assert torch.equal(plane.cell_area(), expected_area)


def test_extent():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    expected_extent = torch.tensor(spacing, dtype=torch.double) * (
        torch.tensor(shape, dtype=torch.double) - 1
    )
    assert torch.equal(plane.length(True), expected_extent)


def test_bounds():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    half_length = plane.length() / 2
    expected_bounds = torch.tensor(
        [
            offset[0] - half_length[0],
            offset[0] + half_length[0],
            offset[1] - half_length[1],
            offset[1] + half_length[1],
        ]
    )
    assert torch.equal(plane.bounds(), expected_bounds)


def test_meshgrid():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    x, y = plane.meshgrid()
    assert x.shape == shape
    assert y.shape == shape


def test_is_same_geometry():
    pg1 = PlanarGrid((10, 10), 5.0, (1.0, 1.0), (0.0, 0.0))
    pg2 = PlanarGrid((10, 10), 5.0, (1.0, 1.0), (0.0, 0.0))
    pg3 = PlanarGrid((10, 10), 5.0, (2.0, 2.0), (1.0, 1.0))
    assert pg1.is_same_geometry(pg2)
    assert not pg1.is_same_geometry(pg3)


def test_visualize():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    tensor = torch.randn(shape)
    with patch("matplotlib.pyplot.show") as mock_show:
        visual = plane._visualize(tensor, show=True, return_fig=True, show_bounds=True)
        mock_show.assert_called_once()
        assert isinstance(visual, Figure)


def test_repr():
    shape, z, spacing, offset = make_planar_grid()
    plane = PlanarGrid(shape=shape, z=z, spacing=spacing, offset=offset)
    expected_repr = (
        "PlanarGrid(shape=(100, 101), z=1.00e+00, spacing=(1.00e-01, 2.00e-01), offset=(0.00e+00, 0.00e+00))"
    )
    assert repr(plane) == expected_repr
