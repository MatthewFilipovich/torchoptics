import unittest

import matplotlib.pyplot as plt
import torch

import torchoptics
from torchoptics import Field, PolarizedField
from torchoptics.elements import *
from torchoptics.profiles import hermite_gaussian


class TestBeamSplitters(unittest.TestCase):
    def test_beam_splitter(self):
        """Test the BeamSplitter class."""
        shape = 64
        z = 0
        field = torchoptics.Field(torch.ones(3, shape, shape), wavelength=700e-9, spacing=1e-5)
        # Create a 50:50 beam splitter
        bs = BeamSplitter(shape, theta=torch.pi / 4, phi_0=0, phi_r=0, phi_t=0, z=z, spacing=1e-5)
        self.assertEqual(bs.shape, (shape, shape))

        # Send a single field through the beam splitter
        bs_field0, bs_field1 = bs.forward(field)
        self.assertIsInstance(bs_field0, torchoptics.Field)
        self.assertIsInstance(bs_field1, torchoptics.Field)
        self.assertTrue(torch.allclose(bs_field0.intensity(), 0.5 * field.intensity()))
        self.assertTrue(torch.allclose(bs_field1.intensity(), 0.5 * field.intensity()))

        # Send two fields through the beam splitter
        bs_field0, bs_field1 = bs.forward(field, field)
        self.assertIsInstance(bs_field0, torchoptics.Field)
        self.assertIsInstance(bs_field1, torchoptics.Field)
        self.assertTrue(torch.allclose(bs_field0.intensity(), 2 * field.intensity()))
        self.assertTrue(torch.allclose(bs_field1.intensity(), 0 * field.intensity()))

        polarized_field = torchoptics.PolarizedField(
            torch.ones(4, 3, shape, shape), wavelength=700e-9, spacing=1e-5
        )

        # Send a single polarized field through the beam splitter
        bs_polarized_field0, bs_polarized_field1 = bs.forward(polarized_field)
        self.assertIsInstance(bs_polarized_field0, torchoptics.PolarizedField)
        self.assertIsInstance(bs_polarized_field1, torchoptics.PolarizedField)
        self.assertTrue(torch.allclose(bs_polarized_field0.intensity(), 0.5 * polarized_field.intensity()))
        self.assertTrue(torch.allclose(bs_polarized_field1.intensity(), 0.5 * polarized_field.intensity()))
        # Send two polarized fields through the beam splitter
        bs_polarized_field0, bs_polarized_field1 = bs.forward(polarized_field, polarized_field)
        self.assertIsInstance(bs_polarized_field0, torchoptics.PolarizedField)
        self.assertIsInstance(bs_polarized_field1, torchoptics.PolarizedField)
        self.assertTrue(torch.allclose(bs_polarized_field0.intensity(), 2 * polarized_field.intensity()))
        self.assertTrue(torch.allclose(bs_polarized_field1.intensity(), 0 * polarized_field.intensity()))


class TestPolarizingBeamSplitters(unittest.TestCase):
    def test_polarizing_beam_splitter(self):
        """Test the PolarizingBeamSplitter class."""
        shape = 32
        z = 0
        field = PolarizedField(torch.ones(4, 3, shape, shape), wavelength=700e-9, spacing=1e-5)
        field.data[:, 2] = 0  # z polarization
        # Create a 50:50 polarized beam splitter
        bs = PolarizingBeamSplitter(shape, z, spacing=1e-5)
        self.assertEqual(bs.shape, (shape, shape))

        # Send a single polarized field through the beam splitter
        bs_field0, bs_field1 = bs.forward(field)
        self.assertIsInstance(bs_field0, torchoptics.PolarizedField)
        self.assertIsInstance(bs_field1, torchoptics.PolarizedField)
        self.assertTrue(torch.allclose(bs_field0.data[:, 0], field.data[:, 0]))
        self.assertTrue(torch.allclose(bs_field0.data[:, 1], 0 * field.data[:, 0]))
        self.assertTrue(torch.allclose(bs_field1.data[:, 0], 0 * field.data[:, 0]))
        self.assertTrue(torch.allclose(bs_field1.data[:, 1], field.data[:, 1]))

        # Test with z-polarized field
        field = PolarizedField(torch.ones(4, 3, shape, shape), wavelength=700e-9, spacing=1e-5)
        with self.assertRaises(ValueError):
            bs.forward(field)


class TestDetectors(unittest.TestCase):
    def test_hermite_gaussian_orthogonality(self):
        """
        Test that the FieldDetector correctly identifies orthogonality of Hermite-Gaussian modes.
        For two different Hermite-Gaussian modes, the inner product should be 0.
        For the same mode, the inner product should be 1.
        """
        shape = (100, 100)  # Grid dimensions
        spacing = (0.1, 0.1)  # Grid spacing
        radius = 1.0  # Radius of the aperture

        # Assuming you have a function to generate Hermite-Gaussian modes
        hg_00 = hermite_gaussian(shape, 0, 0, radius, spacing=spacing)
        hg_10 = hermite_gaussian(shape, 1, 0, radius, spacing=spacing)
        hg_01 = hermite_gaussian(shape, 0, 1, radius, spacing=spacing)
        hg_modes = torch.stack([hg_00, hg_10, hg_01])

        # Create FieldDetector with field_1 as the weight
        field_01 = Field(hg_01, wavelength=1, spacing=spacing)
        detector = FieldDetector(weight=hg_modes, spacing=spacing)

        output = detector(field_01)
        self.assertIsInstance(output, torch.Tensor)
        # The inner product of the same mode should be 1 and different modes should be 0
        self.assertTrue(torch.allclose(output, torch.tensor((0.0, 0.0, 1.0), dtype=torch.double)))

    def test_intensity_detector(self):
        """Test the IntensityDetector class."""
        shape = (100, 100)
        spacing = 1
        field = Field(torch.ones(*shape), wavelength=700e-9, spacing=spacing)
        weight = torch.zeros(2, *shape)
        weight[0, :50, :60] = 1
        weight[1, :40, :30] = 1

        detector = IntensityDetector(weight, spacing=spacing)
        output = detector(field)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(output.shape == (2,))
        self.assertTrue(torch.allclose(output, torch.tensor([3000.0, 1200.0], dtype=torch.double)))
        fig = detector.visualize(0, show=False, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)

        with self.assertRaises(TypeError):
            IntensityDetector("not a tensor", spacing=spacing)
        with self.assertRaises(ValueError):
            IntensityDetector(torch.rand(1, 2, 3, 4), spacing=spacing)


class TestModulatorClasses(unittest.TestCase):
    def setUp(self):
        self.complex_tensor = torch.rand((10, 12), dtype=torch.cdouble)
        self.phase_profile = torch.rand((10, 12))
        self.amplitude_profile = torch.rand((10, 12))
        self.z = 1.5
        self.field = Field(torch.ones(3, 10, 12), wavelength=700e-9, z=self.z, spacing=1)
        torchoptics.set_default_spacing(1)

    def test_modulator_initialization(self):
        modulator = Modulator(self.complex_tensor, self.z)
        self.assertIsInstance(modulator(self.field), Field)
        self.assertTrue(torch.equal(modulator.modulation_profile(), self.complex_tensor))

    def test_phase_modulator_initialization_and_profile(self):
        phase_modulator = PhaseModulator(self.phase_profile, self.z)
        expected_profile = torch.exp(1j * self.phase_profile).cdouble()
        self.assertTrue(torch.allclose(phase_modulator.modulation_profile(), expected_profile))
        self.assertIsInstance(phase_modulator(self.field), Field)

    def test_amplitude_modulator_initialization_and_profile(self):
        amplitude_modulator = AmplitudeModulator(self.amplitude_profile, self.z)
        expected_profile = self.amplitude_profile.cdouble()
        self.assertTrue(torch.allclose(amplitude_modulator.modulation_profile(), expected_profile))
        self.assertIsInstance(amplitude_modulator(self.field), Field)

    def test_phase_modulation_profile_consistency(self):
        phase_modulator = PhaseModulator(self.phase_profile, self.z)
        modulator = Modulator(torch.exp(1j * self.phase_profile), self.z)
        self.assertTrue(torch.allclose(modulator.modulation_profile(), phase_modulator.modulation_profile()))

    def test_polychromatic_phase_modulator(self):
        optical_path_length = torch.rand((10, 12), dtype=torch.double)
        polychromatic_modulator = PolychromaticPhaseModulator(optical_path_length, self.z)
        wavelength = 700e-9
        expected_profile = torch.exp(2j * torch.pi / wavelength * optical_path_length)
        self.assertTrue(
            torch.allclose(polychromatic_modulator.modulation_profile(wavelength), expected_profile)
        )
        self.assertIsInstance(polychromatic_modulator(self.field), Field)

    def test_amplitude_modulation_profile_consistency(self):
        amplitude_modulator = AmplitudeModulator(self.amplitude_profile, self.z)
        modulator = Modulator(self.amplitude_profile.cdouble(), self.z)
        self.assertTrue(
            torch.allclose(modulator.modulation_profile(), amplitude_modulator.modulation_profile())
        )

    def test_error_on_invalid_tensor_input(self):
        with self.assertRaises(TypeError):
            Modulator("not a tensor", self.z)
        with self.assertRaises(TypeError):
            PhaseModulator("not a tensor", self.z)
        with self.assertRaises(TypeError):
            AmplitudeModulator("not a tensor", self.z)

    def test_error_on_incorrect_dimensions(self):
        invalid_tensor = torch.rand((10, 10, 10))  # 3D tensor, should be 2D
        with self.assertRaises(ValueError):
            Modulator(invalid_tensor, self.z)
        with self.assertRaises(ValueError):
            PhaseModulator(invalid_tensor, self.z)
        with self.assertRaises(ValueError):
            AmplitudeModulator(invalid_tensor, self.z)

    def test_visualization(self):
        modulator = Modulator(self.complex_tensor, self.z)
        fig = modulator.visualize(show=False, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_polychromatic_visualization(self):
        optical_path_length = torch.rand((10, 12), dtype=torch.double)
        polychromatic_modulator = PolychromaticPhaseModulator(optical_path_length, self.z)
        fig = polychromatic_modulator.visualize(700e-9, show=False, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)


class TestPolarizedModulatorClasses(unittest.TestCase):
    def setUp(self):
        self.polarized_modulation_profile = torch.rand((3, 3, 10, 12), dtype=torch.cdouble)
        self.phase_profile = torch.rand((3, 3, 10, 12))
        self.amplitude_profile = torch.rand((3, 3, 10, 12))
        self.z = 1.5
        self.polarized_field = PolarizedField(
            torch.ones(4, 3, 10, 12), wavelength=700e-9, z=self.z, spacing=1
        )
        torchoptics.set_default_spacing(1)

    def test_polarized_modulator_initialization(self):
        modulator = PolarizedModulator(self.polarized_modulation_profile, self.z)
        self.assertIsInstance(modulator(self.polarized_field), PolarizedField)
        self.assertTrue(
            torch.equal(modulator.polarized_modulation_profile(), self.polarized_modulation_profile)
        )

    def test_polarized_phase_modulator_initialization_and_profile(self):
        phase_modulator = PolarizedPhaseModulator(self.phase_profile, self.z)
        expected_profile = torch.exp(1j * self.phase_profile).to(dtype=torch.cdouble)
        self.assertTrue(torch.allclose(phase_modulator.polarized_modulation_profile(), expected_profile))
        self.assertIsInstance(phase_modulator(self.polarized_field), PolarizedField)

    def test_polarized_amplitude_modulator_initialization_and_profile(self):
        amplitude_modulator = PolarizedAmplitudeModulator(self.amplitude_profile, self.z)
        expected_profile = self.amplitude_profile.cdouble()
        self.assertTrue(torch.allclose(amplitude_modulator.polarized_modulation_profile(), expected_profile))
        self.assertIsInstance(amplitude_modulator(self.polarized_field), PolarizedField)

    def test_phase_modulation_profile_consistency(self):
        phase_modulator = PolarizedPhaseModulator(self.phase_profile, self.z)
        modulator = PolarizedModulator(torch.exp(1j * self.phase_profile), self.z)
        self.assertTrue(
            torch.allclose(
                modulator.polarized_modulation_profile(), phase_modulator.polarized_modulation_profile()
            )
        )

    def test_amplitude_modulation_profile_consistency(self):
        amplitude_modulator = PolarizedAmplitudeModulator(self.amplitude_profile, self.z)
        modulator = PolarizedModulator(self.amplitude_profile.cdouble(), self.z)
        self.assertTrue(
            torch.allclose(
                modulator.polarized_modulation_profile(), amplitude_modulator.polarized_modulation_profile()
            )
        )

    def test_error_on_invalid_tensor_input(self):
        with self.assertRaises(TypeError):
            PolarizedModulator("not a tensor", self.z)
        with self.assertRaises(TypeError):
            PolarizedPhaseModulator("not a tensor", self.z)
        with self.assertRaises(TypeError):
            PolarizedAmplitudeModulator("not a tensor", self.z)

    def test_error_on_incorrect_dimensions(self):
        with self.assertRaises(ValueError):
            PolarizedModulator(torch.rand((3, 3, 10)), self.z)
        with self.assertRaises(ValueError):
            PolarizedModulator(torch.rand((3, 4, 10, 10)), self.z)

    def test_visualization(self):
        modulator = PolarizedModulator(self.polarized_modulation_profile, self.z)
        fig = modulator.visualize(0, 0, show=False, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)


class TestPolarizers(unittest.TestCase):
    def test_linear_polarizer(self):
        """Test the LinearPolarizer class."""
        shape = (32, 32)
        theta = torch.tensor(torch.pi / 4)
        spacing = 1
        polarizer = LinearPolarizer(shape, theta, spacing=spacing)
        self.assertEqual(polarizer.shape, shape)
        polarized_modulation_profile = polarizer.polarized_modulation_profile()
        expected_matrix = (
            torch.tensor(
                [
                    [torch.cos(theta) ** 2, torch.cos(theta) * torch.sin(theta)],
                    [torch.cos(theta) * torch.sin(theta), torch.sin(theta) ** 2],
                ],
                dtype=torch.cdouble,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(2, 2, *shape)
        )
        self.assertTrue(torch.allclose(polarized_modulation_profile[:2, :2], expected_matrix))
        field = PolarizedField(torch.ones(4, 3, *shape), wavelength=700e-9, spacing=spacing)
        output_field = polarizer(field)
        self.assertIsInstance(output_field, torchoptics.PolarizedField)

    def test_left_circular_polarizer(self):
        """Test the LeftCircularPolarizer class."""
        shape = (32, 32)
        spacing = 1
        polarizer = LeftCircularPolarizer(shape, spacing=spacing)
        self.assertEqual(polarizer.shape, shape)
        polarization_modulation_profile = polarizer.polarized_modulation_profile()
        expected_matrix = (
            torch.tensor([[0.5, -0.5j], [0.5j, 0.5]], dtype=torch.cdouble)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(2, 2, *shape)
        )
        self.assertTrue(torch.allclose(polarization_modulation_profile[:2, :2], expected_matrix))
        field = PolarizedField(torch.ones(4, 3, *shape), wavelength=700e-9, spacing=spacing)
        output_field = polarizer(field)
        self.assertIsInstance(output_field, torchoptics.PolarizedField)

    def test_right_circular_polarizer(self):
        """Test the RightCircularPolarizer class."""
        shape = (32, 32)
        spacing = 1
        polarizer = RightCircularPolarizer(shape, spacing=spacing)
        self.assertEqual(polarizer.shape, shape)
        field = PolarizedField(torch.ones(4, 3, *shape), wavelength=700e-9, spacing=spacing)
        output_field = polarizer(field)
        self.assertIsInstance(output_field, torchoptics.PolarizedField)


class TestLens(unittest.TestCase):
    def test_len(self):
        shape = (64, 64)
        focal_length = 50.0
        wavelength = 500e-9
        spacing = 1e-5
        lens = Lens(shape, focal_length, 0, spacing)
        self.assertEqual(lens.shape, shape)
        self.assertEqual(lens.focal_length, focal_length)
        self.assertTrue(lens.is_circular_lens)
        with self.assertRaises(TypeError):
            lens.is_circular_lens = "not a bool"
        self.assertTrue(lens.modulation_profile(wavelength).dtype == torch.cdouble)
        self.assertTrue("is_circular_lens=True" in repr(lens))
        field = Field(torch.ones(3, *shape), wavelength=wavelength, spacing=spacing)
        output_field = lens(field)
        self.assertIsInstance(output_field, torchoptics.Field)


class TestWaveplates(unittest.TestCase):

    def test_waveplate_forward(self):
        """Test the forward method of the Waveplate."""
        shape = (32, 32)
        phi = torch.tensor(torch.pi / 2)
        theta = torch.tensor(torch.pi / 4)
        spacing = 1
        waveplate = Waveplate(shape, phi, theta, spacing=spacing)
        field = PolarizedField(torch.ones(4, 3, *shape), wavelength=700e-9, spacing=spacing)
        output_field = waveplate(field)
        self.assertIsInstance(output_field, torchoptics.PolarizedField)

    def test_waveplate_modulation_profile(self):
        """Test the polarized matrix of the Waveplate with theta and phi set to 0."""
        shape = (32, 32)
        phi = torch.tensor(0.0)
        theta = torch.tensor(0.0)
        spacing = 1
        waveplate = Waveplate(shape, phi, theta, spacing=spacing)
        polarized_modulation_profile = waveplate.polarized_modulation_profile()
        expected_matrix = (
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.cdouble,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(2, 2, *shape)
        )
        self.assertTrue(torch.allclose(polarized_modulation_profile[:2, :2], expected_matrix))

    def test_waveplate_profile(self):
        """Test the polarization matrix of the Waveplate for a quarter-wave plate (QWP)."""
        shape = (32, 30)
        phi = torch.tensor(torch.pi / 2)
        theta = torch.tensor(torch.pi / 4)
        spacing = 1
        waveplate = Waveplate(shape, phi, theta, spacing=spacing)
        polarized_modulation_profile = waveplate.polarized_modulation_profile()
        expected_matrix = 0.5 * torch.tensor(
            [
                [1 + 1j, 1 - 1j, 0],
                [1 - 1j, 1 + 1j, 0],
                [0, 0, 2],
            ],
            dtype=torch.cdouble,
        ).unsqueeze(-1).unsqueeze(-1).expand(3, 3, *shape)
        self.assertTrue(torch.allclose(polarized_modulation_profile, expected_matrix))

    def test_quarter_waveplate_profile(self):
        """Test the polarization matrix of the QuarterWaveplate."""
        shape = (32, 32)
        theta = torch.tensor(torch.pi / 4)
        spacing = 1
        qwp = QuarterWaveplate(shape, theta, spacing=spacing)
        waveplate = Waveplate(shape, torch.pi / 2, theta, spacing=spacing)
        self.assertTrue(
            torch.allclose(qwp.polarized_modulation_profile(), waveplate.polarized_modulation_profile())
        )

    def test_half_waveplate_profile(self):
        """Test the polarization matrix of the HalfWaveplate."""
        shape = (32, 32)
        theta = torch.tensor(torch.pi / 4)
        spacing = 1
        hwp = HalfWaveplate(shape, theta, spacing=spacing)
        waveplate = Waveplate(shape, torch.pi, theta, spacing=spacing)
        self.assertTrue(
            torch.allclose(hwp.polarized_modulation_profile(), waveplate.polarized_modulation_profile())
        )


class TestElement(unittest.TestCase):

    def test_element(self):
        shape = (32, 32)
        z = 0
        spacing = 1
        element = Element(shape, z, spacing=spacing)
        with self.assertRaises(TypeError):
            element.validate_field("not a field")
        field = Field(torch.ones(3, *shape), wavelength=700e-9, spacing=spacing, z=2)
        with self.assertRaises(ValueError):
            element.validate_field(field)
