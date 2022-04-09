import torch
import torch.nn as nn

from torch.nn.functional import softmax

from src.managers.base import NamedObject, ObjectManager


class WeightedMeanFuser(nn.Module, NamedObject):
    name = 'weighted_mean'
    def __init__(self, config):
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(config.n_modality) / config.n_modality,
            requires_grad=True
        )
    
    
    def forward(self, latents):
        weights = softmax(self.weights, dim=0)
        weighted_latents = [parameter * latent for parameter, latent in zip(weights, latents)]
        return torch.sum(torch.stack(weighted_latents, -1), dim=-1)


class FuserManager(nn.Module, ObjectManager):
    """\
    Fuser
    """
    name = 'fusers'
    constructors = [WeightedMeanFuser]
    
    def __init__(self, config):
        super().__init__()
        self.layer = FuserManager.get_constructor_by_name(config.method)(config)

    def forward(self, x):
        return self.layer(x)