from torch.nn import Module, ReLU, Sigmoid, Softmax, Tanh

from src.managers.base import NamedObject, ObjectManager


class ReLUActivation(ReLU, NamedObject):
    name = 'relu'


class SigmoidActivation(Sigmoid, NamedObject):
    name = 'sigmoid'


class SoftmaxActivation(Softmax, NamedObject):
    name = 'softmax'


class TanhActivation(Tanh, NamedObject):
    name = 'tanh'


class ActivationManager(Module, ObjectManager):
    """\
    Activation
    """
    name = 'activations'
    constructors = [ReLUActivation, SigmoidActivation, SoftmaxActivation, TanhActivation]