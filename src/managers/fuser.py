import torch

from torch.nn import Module

from src.managers.base import NamedObject, ObjectManager


class MeanFuser(Module, NamedObject):
    """\
    MeanFuser
    """
    name = 'mean'

    def __init__(self):
        super().__init__()
    

    def forward(self, x):
        return torch.mean(torch.stack(x, -1), dim=-1)


class FuserManager(Module, ObjectManager):
    """\
    Fuser
    """
    name = 'fusers'
    constructors = [MeanFuser]
    
    def __init__(self, method):
        super().__init__()
        self.layer = FuserManager.get_constructor_by_name(method)()

    def forward(self, x):
        return self.layer(x)