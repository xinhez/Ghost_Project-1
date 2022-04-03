import torch

from torch.nn import Module, ModuleList

from config import create_model_config_from_data
from managers.fuser import FuserManager
from models.labelEncoder import LabelEncoder
from models.mlp import MLP


class Model(Module):
    """
    Model
    """
    name = 'Model'
    def __init__(self, config):
        super().__init__()
        self.data_label_encoder = LabelEncoder() 
        self.data_batch_encoder = LabelEncoder()

        self.encoders       = create_module_list(MLP,          config.encoders)
        self.decoders       = create_module_list(MLP,          config.decoders)
        self.discriminators = create_module_list(MLP,          config.discriminators)
        self.fusers         = create_module_list(FuserManager, config.fusion_methods)
        self.clusters       = create_module_list(MLP,          config.clusters)

        # self.optimizers


    def forward(self, modalities):
        self.modalities = modalities 

        self.latents = [
            encoder(modality) for (encoder, modality) in zip(self.encoders, modalities)
        ]

        self.translations = [
            [decoder(latent) for latent in self.latents] for decoder in self.decoders
        ]

        self.fused_latents = [
            fuser(self.latents) for fuser in self.fusers
        ]

        self.cluster_outputs = [
            cluster(fused_latent) for (cluster, fused_latent) in zip(self.clusters, self.fused_latents)
        ]

        self.discriminator_real_outputs = [
            discriminator(modality) for (discriminator, modality) in zip(self.discriminators, modalities)
        ]

        self.discriminator_fake_outputs = [
            discriminator(self.translations[i][i]) for i, discriminator in enumerate(self.discriminators)
        ]

        return self.latents, self.fused_latents, self.translations, self.cluster_outputs


    def save_model(self, path):
        """\
        Save current model parameters to the given path.

        path
            The absolute path to the desired location.
        """
        torch.save(self, path)


# ==================== Model Generator ====================
def create_module_list(constructor, configs):
    """\
    Create a list of modules using the given constructor of the given configs.
    """
    return ModuleList([constructor(config) for config in configs])


def create_model_from_data(data):
    """\
    Create a new model based on the given dataset.
    """
    return Model(create_model_config_from_data(data))


def load_model_from_path(path):
    """\
    Create a new model with model parameters from the given path.

    path
        The absolute path to the desired location.
    """
    return torch.load(path)