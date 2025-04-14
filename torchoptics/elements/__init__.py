"""This module contains the classes for the optical elements."""

from .beam_splitters import BeamSplitter, PolarizingBeamSplitter
from .detectors import Detector, LinearDetector
from .elements import Element, ModulationElement, PolarizedModulationElement, PolychromaticModulationElement
from .identity_element import IdentityElement
from .lens import CylindricalLens, Lens
from .modulators import AmplitudeModulator, Modulator, PhaseModulator, PolychromaticPhaseModulator
from .polarized_modulators import PolarizedAmplitudeModulator, PolarizedModulator, PolarizedPhaseModulator
from .polarizers import LeftCircularPolarizer, LinearPolarizer, RightCircularPolarizer
from .waveplates import HalfWaveplate, QuarterWaveplate, Waveplate
