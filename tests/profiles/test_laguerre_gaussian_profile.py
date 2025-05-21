import torch

from torchoptics.profiles import gaussian, laguerre_gaussian


def make_laguerre_gaussian_profiles():
    shape = (300, 300)
    wavelength = 1
    spacing = 1
    offset = (0.0, 0.0)
    waist_radius = 40.0
    z = 1
    return [
        laguerre_gaussian(
            shape=shape,
            p=p,
            l=l,
            waist_radius=waist_radius,
            wavelength=wavelength,
            waist_z=z,
            spacing=spacing,
            offset=offset,
        )
        for p in range(3)
        for l in range(-2, 3)  # noqa: E741
        if p + abs(l) < 3
    ]


def test_laguerre_gaussian_orthogonality() -> None:
    profiles = make_laguerre_gaussian_profiles()
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            inner_product = torch.sum(profiles[i].conj() * profiles[j]).abs().item()
            assert abs(inner_product) < 1e-7


def test_laguerre_gaussian_dtype() -> None:
    profiles = make_laguerre_gaussian_profiles()
    for profile in profiles:
        assert profile.dtype == torch.cdouble


def test_laguerre_gaussian_normalization() -> None:
    profiles = make_laguerre_gaussian_profiles()
    for profile in profiles:
        inner_product = torch.sum(profile.conj() * profile).abs().item()
        assert abs(inner_product - 1.0) < 1e-7


def test_laguerre_gaussian_equivalence() -> None:
    shape = (300, 300)
    wavelength = 1
    spacing = 1
    offset = (0.0, 0.0)
    waist_radius = 40.0
    laguerre_gaussian_profile = laguerre_gaussian(
        shape=shape,
        p=0,
        l=0,
        waist_radius=waist_radius,
        wavelength=wavelength,
        waist_z=0,
        spacing=spacing,
        offset=offset,
    )
    gaussian_profile = gaussian(
        shape=shape,
        waist_radius=waist_radius,
        wavelength=wavelength,
        spacing=spacing,
        offset=offset,
    )
    assert torch.allclose(laguerre_gaussian_profile, gaussian_profile, atol=1e-5)
