import numpy, random, torch

from torch.nn import Module, ModuleList

from src.config import combine_config
from src.managers.fuser import FuserManager
from src.managers.technique import TechniqueManager
from src.models.labelEncoder import LabelEncoder
from src.models.mlp import MLP
from src.models.optimizer import Optimizer
from src.utils import set_random_seed


class ModuleNames():
    encoders       = 'encoders'
    decoders       = 'decoders'
    discriminators = 'discriminators'
    fusers         = 'fusers'
    clusters       = 'clusters'


class Model(Module):
    """
    Model
    """
    name = 'Model'
    

    def __init__(self, config):
        super().__init__()
        self.config = None
        self.data_label_encoder = LabelEncoder() 
        self.data_batch_encoder = LabelEncoder()
        self.update_config(config)

    
    def update_config(self, config):
        set_random_seed(numpy, random, torch)
        config = combine_config(self.config, config)
        self.config = config

        self.encoders       = create_module_list(MLP,          config.encoders)
        self.decoders       = create_module_list(MLP,          config.decoders)
        self.discriminators = create_module_list(MLP,          config.discriminators)
        self.fusers         = create_module_list(FuserManager, config.fusers)
        self.clusters       = create_module_list(MLP,          config.clusters)
        self.best_head      = 0 

        self.optimizers = {
            ModuleNames.encoders:       create_optimizer_list(config.optimizers.encoders,       self.encoders),
            ModuleNames.decoders:       create_optimizer_list(config.optimizers.decoders,       self.decoders),
            ModuleNames.discriminators: create_optimizer_list(config.optimizers.discriminators, self.discriminators),
            ModuleNames.fusers:         create_optimizer_list(config.optimizers.fusers,         self.fusers),
            ModuleNames.clusters:       create_optimizer_list(config.optimizers.clusters,       self.clusters),
        }
        

    @property
    def n_head(self):
        if len(self.fusers) != len(self.clusters):
            raise Exception("Model must have the same number of fusers and clusters.")
        return len(self.fusers)


    @property
    def n_modality(self):
        return len(self.encoders)

    
    def set_device(self, device):
        self.device = device 
        self.to(device)


    def forward(self, modalities, batches, labels):
        self.modalities = [modality.to(device=self.device) for modality in modalities]
        self.batches    = batches.to(device=self.device)
        self.labels     = labels.to(device=self.device)

        self.latents = [
            encoder(modality) for (encoder, modality) in zip(self.encoders, modalities)
        ]

        self.translations = [
            [
                decoder(latent) for latent in self.latents
            ] for decoder in self.decoders
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

        self.predictions = [
            self.cluster_outputs[head].argmax(axis=1) for head in range(self.n_head)
        ]
        
        return self.translations, self.predictions[self.best_head], self.fused_latents[self.best_head]


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
    technique = TechniqueManager.get_constructor_by_name(data.technique)()
    return Model(technique.get_default_config(data))


def load_model_from_path(path):
    """\
    Create a new model with model parameters from the given path.

    path
        The absolute path to the desired location.
    """
    return torch.load(path)


def create_optimizer_list(configs, modules):
    """\
    Create a list of optimizers using the given configs for the paired given modules.
    """
    return [Optimizer(config, module.parameters()) for config, module in zip(configs, modules)]