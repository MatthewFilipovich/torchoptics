Introduction
============

TorchOptics is an object-oriented library for modeling optical fields and simulating their evolution through optical systems. 
It consists of three core classes: :class:`Field`, :class:`Element`, and :class:`System`. Each class includes methods for simulation and analysis, 
with properties stored as PyTorch tensors that can be dynamically updated with a single line of code. 
These classes inherit from PyTorch's :class:`torch.nn.Module` class to enable standard PyTorch functionalities, 
including device management (CPU and GPU) and parameter registration for gradient-based optimization.