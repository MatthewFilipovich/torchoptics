import torch

from torchoptics.profiles import shapes


def make_shape_args():
    return dict(shape=(100, 100), spacing=(0.1, 0.1), offset=(0.0, 0.0))


def test_checkerboard():
    args = make_shape_args()
    tile_length = (10, 10)
    num_tiles = (10, 10)
    pattern = shapes.checkerboard(tile_length=tile_length, num_tiles=num_tiles, **args)
    assert pattern.shape == args["shape"]
    assert torch.all((pattern == 0) | (pattern == 1))
    assert pattern.dtype == torch.double


def test_circle():
    args = make_shape_args()
    radius = 5.0
    profile = shapes.circle(radius=radius, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
    assert profile.dtype == torch.double


def test_rectangle():
    args = make_shape_args()
    side = (10, 20)
    profile = shapes.rectangle(side=side, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
    assert profile.dtype == torch.double


def test_square():
    args = make_shape_args()
    side = 10.0
    profile = shapes.square(side=side, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
    assert profile.dtype == torch.double


def test_triangle():
    args = make_shape_args()
    base = 10.0
    height = 20.0
    profile = shapes.triangle(base=base, height=height, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
    assert profile.dtype == torch.double
