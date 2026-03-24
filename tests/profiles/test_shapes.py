import math

import torch

from torchoptics.profiles import shapes


def make_shape_args():
    return {"shape": (100, 100), "spacing": (0.1, 0.1), "offset": (0.0, 0.0)}


def test_checkerboard():
    args = make_shape_args()
    tile_length = (10, 10)
    num_tiles = (10, 10)
    pattern = shapes.checkerboard(tile_length=tile_length, num_tiles=num_tiles, **args)
    assert pattern.shape == args["shape"]
    assert torch.all((pattern == 0) | (pattern == 1))


def test_circle():
    args = make_shape_args()
    radius = 5.0
    profile = shapes.circle(radius=radius, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))


def test_hexagon():
    args = make_shape_args()
    radius = 4.0
    profile = shapes.hexagon(radius=radius, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
    # Center point should be inside the hexagon
    assert profile[50, 50] == 1.0
    # hexagon should equal regular_polygon with 6 sides
    expected = shapes.regular_polygon(num_sides=6, radius=radius, **args)
    assert torch.equal(profile, expected)


def test_octagon():
    args = make_shape_args()
    radius = 4.0
    profile = shapes.octagon(radius=radius, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
    # Center point should be inside the octagon
    assert profile[50, 50] == 1.0
    # octagon should equal regular_polygon with 8 sides and theta=pi/8
    expected = shapes.regular_polygon(num_sides=8, radius=radius, theta=math.pi / 8, **args)
    assert torch.equal(profile, expected)


def test_regular_polygon_output_shape():
    args = make_shape_args()
    for num_sides in [3, 5, 6, 8, 12]:
        profile = shapes.regular_polygon(num_sides=num_sides, radius=4.0, **args)
        assert profile.shape == args["shape"]
        assert torch.all((profile == 0) | (profile == 1))


def test_regular_polygon_center_inside():
    args = make_shape_args()
    for num_sides in [3, 4, 5, 6, 8]:
        profile = shapes.regular_polygon(num_sides=num_sides, radius=4.0, **args)
        assert profile[50, 50] == 1.0


def test_regular_polygon_rotation():
    args = make_shape_args()
    # Rotating by a full period (2*pi / num_sides) should give the same polygon
    num_sides = 6
    radius = 4.0
    profile = shapes.regular_polygon(num_sides=num_sides, radius=radius, **args)
    rotated = shapes.regular_polygon(
        num_sides=num_sides, radius=radius, theta=2 * math.pi / num_sides, **args
    )
    assert torch.equal(profile, rotated)


def test_rectangle():
    args = make_shape_args()
    side = (10, 20)
    profile = shapes.rectangle(side=side, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))


def test_square():
    args = make_shape_args()
    side = 10.0
    profile = shapes.square(side=side, **args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
    # Center should be inside
    assert profile[50, 50] == 1.0
    # square should equal rectangle with equal sides
    expected = shapes.rectangle(side=(side, side), **args)
    assert torch.equal(profile, expected)


def test_triangle():
    args = make_shape_args()
    base = 10.0
    height = 20.0
    updated_args = {**args, "theta": 0.0}
    profile = shapes.triangle(base=base, height=height, **updated_args)
    assert profile.shape == args["shape"]
    assert torch.all((profile == 0) | (profile == 1))
