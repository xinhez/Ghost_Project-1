import torch.nn as nn

from src.managers.base import NamedObject, ObjectManager


class ReLUActivation(nn.ReLU, NamedObject):
    name = 'relu'


class SigmoidActivation(nn.Sigmoid, NamedObject):
    name = 'sigmoid'


class SoftmaxActivation(nn.Softmax, NamedObject):
    name = 'softmax'


class TanhActivation(nn.Tanh, NamedObject):
    name = 'tanh'


class ActivationManager(nn.Module, ObjectManager):
    """\
    Activation
    """
    name = 'activations'
    constructors = [ReLUActivation, SigmoidActivation, SoftmaxActivation, TanhActivation]