from torch.nn import Module, ReLU, Sigmoid, Softmax, Tanh

from utils import NamedObject, ObjectManager


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
    constructors = [ReLUActivation, SigmoidActivation, SoftmaxActivation, TanhActivation]


    def __init__(self, method):
        super().__init__()
        self.layer = ActivationManager.get_constructor_by_name(method)()
    

    def forward(self, x):
        return self.layer(x)