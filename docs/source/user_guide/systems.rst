Systems
========

The :class:`~torchoptics.System` class models an optical system as a sequence of :class:`~torchoptics.elements.Element` objects arranged along the optical axis. 
Its :meth:`~torchoptics.System.forward()` method calculates the evolution of an input field as it propagates through the system and returns the field after it has been processed by the final element. Additionally, the :meth:`~torchoptics.System.measure()` method computes the field at any specified position within the system.
