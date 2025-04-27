import torch

from torchoptics.profiles import gaussian, hermite_gaussian


def make_hermite_gaussian_profiles():
    shape = (300, 300)
    wavelength = 1
    spacing = 1
    offset = (0.0, 0.0)
    waist_radius = 40.0
    z = 1
    profiles = [
        hermite_gaussian(
            shape=shape,
            m=m,
            n=n,
            waist_z=z,
            waist_radius=waist_radius,
            wavelength=wavelength,
            spacing=spacing,
            offset=offset,
        )
        for m in range(3)
        for n in range(3)
        if m + n < 3
    ]
    return profiles


def test_hermite_gaussian_orthogonality():
    profiles = make_hermite_gaussian_profiles()
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            inner_product = torch.sum(profiles[i].conj() * profiles[j]).abs().item()
            assert abs(inner_product) < 1e-7


def test_hermite_gaussian_dtype():
    profiles = make_hermite_gaussian_profiles()
    for profile in profiles:
        assert profile.dtype == torch.cdouble


def test_hermite_gaussian_normalization():
    profiles = make_hermite_gaussian_profiles()
    for profile in profiles:
        inner_product = torch.sum(profile.conj() * profile).abs().item()
        assert abs(inner_product - 1.0) < 1e-7


def test_hermite_gaussian_equivalence():
    shape = (300, 300)
    wavelength = 1
    spacing = 1
    offset = (0.0, 0.0)
    waist_radius = 40.0
    hermite_gaussian_profile = hermite_gaussian(
        shape=shape,
        m=0,
        n=0,
        waist_radius=waist_radius,
        wavelength=wavelength,
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
    assert torch.allclose(hermite_gaussian_profile, gaussian_profile, atol=1e-5)
