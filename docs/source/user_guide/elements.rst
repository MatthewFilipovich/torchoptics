Elements
=========

Optical elements are implemented as subclasses of the :class:`~torchoptics.elements.Element` class and are modeled as  planar surfaces in accordance with Fourier optics. TorchOptics includes several element types, such as modulators, detectors, and beam splitters. 
Each element applies a transformation to an input field defined in its ``forward()`` method. For instance, the :class:`~torchoptics.elements.Lens` class, which models a thin lens, applies a quadratic phase factor to the field's wavefront.
