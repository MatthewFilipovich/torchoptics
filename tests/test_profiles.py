import unittest

import torch

from torchoptics import CoherenceField, Field
from torchoptics.profiles import *


class TestLensProfile(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 100)
        self.focal_length = 50.0
        self.wavelength = 0.5
        self.spacing = (0.1, 0.1)
        self.offset = (0.0, 0.0)
        self.is_circular_lens = True

        self.phase_profile = lens(
            shape=self.shape,
            focal_length=self.focal_length,
            wavelength=self.wavelength,
            spacing=self.spacing,
            offset=self.offset,
            is_circular_lens=self.is_circular_lens,
        )

    def test_lens(self):
        self.assertEqual(self.phase_profile.shape, self.shape)
        self.assertTrue(torch.is_complex(self.phase_profile))
        self.assertAlmostEqual(
            self.phase_profile[self.shape[0] // 2, self.shape[1] // 2].abs().item(), 1.0, places=5
        )

    def test_circular_lens_mask(self):
        radial_dist = torch.sqrt(
            torch.square(torch.linspace(-5, 5, self.shape[0]).unsqueeze(1))
            + torch.square(torch.linspace(-5, 5, self.shape[1]).unsqueeze(0))
        )
        lens_diameter = min(self.shape) * self.spacing[0]
        mask = radial_dist > lens_diameter / 2

        self.assertTrue(torch.all(self.phase_profile[mask] == 0))


class TestHermiteGaussianProfile(unittest.TestCase):
    def setUp(self):
        self.shape = (300, 300)
        self.wavelength = 1
        self.spacing = 1
        self.offset = (0.0, 0.0)
        self.waist_radius = 40.0
        self.z = 1

        self.profiles = [
            hermite_gaussian(
                shape=self.shape,
                m=m,
                n=n,
                waist_z=self.z,
                waist_radius=self.waist_radius,
                wavelength=self.wavelength,
                spacing=self.spacing,
                offset=self.offset,
            )
            for m in range(3)
            for n in range(3)
            if m + n < 3
        ]

    def test_orthogonality(self):
        for i in range(len(self.profiles)):
            for j in range(i + 1, len(self.profiles)):
                inner_product = torch.sum(self.profiles[i].conj() * self.profiles[j]).abs().item()
                self.assertAlmostEqual(inner_product, 0.0)

    def test_normalization(self):
        for profile in self.profiles:
            inner_product = torch.sum(profile.conj() * profile).abs().item()
            self.assertAlmostEqual(inner_product, 1.0)

    def test_gaussian_equivalence(self):
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

        self.assertTrue(torch.allclose(hermite_gaussian_profile, gaussian_profile, atol=1e-5))


class TestLaguerreGaussianProfile(unittest.TestCase):
    def setUp(self):
        self.shape = (300, 300)
        self.wavelength = 1
        self.spacing = 1
        self.offset = (0.0, 0.0)
        self.waist_radius = 40.0
        self.z = 1

        self.profiles = [
            laguerre_gaussian(
                shape=self.shape,
                p=p,
                l=l,
                waist_radius=self.waist_radius,
                wavelength=self.wavelength,
                waist_z=self.z,
                spacing=self.spacing,
                offset=self.offset,
            )
            for p in range(3)
            for l in range(-2, 3)
            if p + abs(l) < 3
        ]

    def test_orthogonality(self):
        for i in range(len(self.profiles)):
            for j in range(i + 1, len(self.profiles)):
                inner_product = torch.sum(self.profiles[i].conj() * self.profiles[j]).abs().item()
                self.assertAlmostEqual(inner_product, 0.0)

    def test_normalization(self):
        for profile in self.profiles:
            inner_product = torch.sum(profile.conj() * profile).abs().item()
            self.assertAlmostEqual(inner_product, 1.0)

    def test_gaussian_equivalence(self):
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

        self.assertTrue(torch.allclose(laguerre_gaussian_profile, gaussian_profile, atol=1e-5))


class TestShapes(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 100)
        self.spacing = (0.1, 0.1)
        self.offset = (0.0, 0.0)

    def test_checkerboard(self):
        tile_length = (10, 10)
        num_tiles = (10, 10)
        pattern = shapes.checkerboard(
            shape=self.shape,
            tile_length=tile_length,
            num_tiles=num_tiles,
            spacing=self.spacing,
            offset=self.offset,
        )
        self.assertEqual(pattern.shape, self.shape)
        self.assertTrue(torch.all((pattern == 0) | (pattern == 1)))

    def test_circle(self):
        radius = 5.0
        profile = shapes.circle(
            shape=self.shape,
            radius=radius,
            spacing=self.spacing,
            offset=self.offset,
        )
        self.assertEqual(profile.shape, self.shape)
        self.assertTrue(torch.all((profile == 0) | (profile == 1)))

    def test_rectangle(self):
        side = (10, 20)
        profile = shapes.rectangle(
            shape=self.shape,
            side=side,
            spacing=self.spacing,
            offset=self.offset,
        )
        self.assertEqual(profile.shape, self.shape)
        self.assertTrue(torch.all((profile == 0) | (profile == 1)))

    def test_square(self):
        side = 10.0
        profile = shapes.square(
            shape=self.shape,
            side=side,
            spacing=self.spacing,
            offset=self.offset,
        )
        self.assertEqual(profile.shape, self.shape)
        self.assertTrue(torch.all((profile == 0) | (profile == 1)))


class TestGratings(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 100)
        self.spacing = (0.1, 0.1)
        self.offset = (0.0, 0.0)
        self.theta = 0.0

    def test_blazed_grating(self):
        period = 10.0
        profile = gratings.blazed_grating(
            shape=self.shape,
            period=period,
            spacing=self.spacing,
            offset=self.offset,
            theta=self.theta,
        )
        self.assertEqual(profile.shape, self.shape)
        self.assertTrue(torch.is_complex(profile))

    def test_sinusoidal_amplitude_grating(self):
        period = 10.0
        m = 0.5
        profile = gratings.sinusoidal_amplitude_grating(
            shape=self.shape,
            m=m,
            period=period,
            spacing=self.spacing,
            offset=self.offset,
            theta=self.theta,
        )
        self.assertEqual(profile.shape, self.shape)
        self.assertTrue(torch.all((profile >= 0) & (profile <= 1)))

    def test_sinusoidal_phase_grating(self):
        period = 10.0
        m = 0.5
        profile = gratings.sinusoidal_phase_grating(
            shape=self.shape,
            m=m,
            period=period,
            spacing=self.spacing,
            offset=self.offset,
            theta=self.theta,
        )
        self.assertEqual(profile.shape, self.shape)
        self.assertTrue(torch.is_complex(profile))


class TestSpecialProfiles(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 100)
        self.spacing = (0.1, 0.1)
        self.offset = (0.0, 0.0)

    def test_airy(self):
        scale = 10.0
        profile = special.airy(
            shape=self.shape,
            scale=scale,
            spacing=self.spacing,
            offset=self.offset,
        )
        self.assertEqual(profile.shape, self.shape)
        self.assertTrue(torch.all(profile >= 0))

    def test_sinc(self):
        scale = (10.0, 20.0)
        profile = special.sinc(
            shape=self.shape,
            scale=scale,
            spacing=self.spacing,
            offset=self.offset,
        )
        self.assertEqual(profile.shape, self.shape)
        self.assertTrue(torch.all(profile >= 0))


class TestGaussianSchellModel(unittest.TestCase):
    def setUp(self):
        self.shape = (10, 15)
        self.waist_radius = 50e-6
        self.coherence_width = torch.inf
        self.spacing = 10e-6
        self.wavelength = 700e-9

    def test_gaussian_schell_model_shape(self):
        coherence_data = gaussian_schell_model(
            shape=self.shape,
            waist_radius=self.waist_radius,
            coherence_width=self.coherence_width,
            spacing=self.spacing,
        )

        self.assertEqual(coherence_data.shape, (10, 15, 10, 15))

    def test_identical_with_gaussian(self):
        coherence_data = gaussian_schell_model(
            shape=self.shape,
            waist_radius=self.waist_radius,
            coherence_width=self.coherence_width,
            spacing=self.spacing,
        )
        gaussian_data = gaussian(
            shape=self.shape,
            waist_radius=self.waist_radius,
            wavelength=1,
            spacing=self.spacing,
        )
        field = Field(gaussian_data, spacing=self.spacing, wavelength=self.wavelength)
        coherence_field = CoherenceField(coherence_data, spacing=self.spacing, wavelength=self.wavelength)

        self.assertTrue(torch.allclose(field.intensity(), coherence_field.intensity()))
        self.assertTrue(
            torch.allclose(
                field.propagate_to_z(0.2).intensity(), coherence_field.propagate_to_z(0.2).intensity()
            )
        )

    def test_incoherent(self):
        incoherent_data = gaussian_schell_model(
            shape=self.shape,
            waist_radius=self.waist_radius,
            coherence_width=0,
            spacing=self.spacing,
        )
        incoherent_data = incoherent_data.view(self.shape[0] * self.shape[1], -1)
        incoherent_data[torch.eye(self.shape[0] * self.shape[1], dtype=bool)] = 0
        self.assertTrue(torch.all(incoherent_data == 0))  # off-diagonal elements should be zero


if __name__ == "__main__":
    unittest.main()