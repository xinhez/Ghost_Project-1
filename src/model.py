import torch
import torch.nn as nn

from itertools import chain

from src.config import combine_config
from src.managers.technique import TechniqueManager
from src.models.fuser import FuserManager
from src.models.labelEncoder import LabelEncoder
from src.models.mlp import MLP
from src.models.optimizer import Optimizer

BEST_HEAD = "best_head"
CONFIG = "config"
DATA_LABEL_ENCODER = "data_label_encoder"
DATA_BATCH_ENCODER = "data_batch_encoder"
WEIGHTS = "weights"


def create_module_list(constructor, configs):
    """\
    Create a list of modules using the given constructor of the given configs.
    """
    return nn.ModuleList([constructor(config) for config in configs])


class ModuleNames:
    encoders = "encoders"
    decoders = "decoders"
    discriminators = "discriminators"
    fusers = "fusers"
    projectors = "projectors"
    clusters = "clusters"


class Model(nn.Module):
    """
    Model
    """

    name = "Model"

    @staticmethod
    def kaiming_init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)

    def __init__(self, config):
        super().__init__()
        self.config = None
        self.data_label_encoder = LabelEncoder()
        self.data_batch_encoder = LabelEncoder()
        self.update_config(config)

    def update_config(self, config):
        config = combine_config(self.config, config)
        self.config = config

        self.modules_by_names = nn.ModuleDict(
            {
                ModuleNames.encoders: create_module_list(MLP, config.encoders),
                ModuleNames.decoders: create_module_list(MLP, config.decoders),
                ModuleNames.discriminators: create_module_list(
                    MLP, config.discriminators
                ),
                ModuleNames.fusers: create_module_list(FuserManager, config.fusers),
                ModuleNames.projectors: create_module_list(MLP, config.projectors),
                ModuleNames.clusters: create_module_list(MLP, config.clusters),
            }
        )

        self.optimizers_by_schedule = {}

        self.register_buffer(BEST_HEAD, torch.tensor(0, dtype=torch.long))

        self.apply(Model.kaiming_init_weights)

    def create_optimizer_for_schedule(
        self, learning_rate, config, schedule_name, module_names
    ):
        if schedule_name not in self.optimizers_by_schedule:
            optimizer = Optimizer(
                learning_rate,
                config,
                chain.from_iterable(
                    [
                        self.modules_by_names[module_name].parameters()
                        for module_name in module_names
                    ]
                ),
            )
            self.optimizers_by_schedule[schedule_name] = optimizer

    @property
    def encoders(self):
        return self.modules_by_names[ModuleNames.encoders]

    @property
    def decoders(self):
        return self.modules_by_names[ModuleNames.decoders]

    @property
    def discriminators(self):
        return self.modules_by_names[ModuleNames.discriminators]

    @property
    def fusers(self):
        return self.modules_by_names[ModuleNames.fusers]

    @property
    def projectors(self):
        return self.modules_by_names[ModuleNames.projectors]

    @property
    def clusters(self):
        return self.modules_by_names[ModuleNames.clusters]

    @property
    def n_head(self):
        if len(self.fusers) != len(self.clusters):
            raise Exception("Model must have the same number of fusers and clusters.")
        return len(self.fusers)

    @property
    def n_modality(self):
        return len(self.encoders)

    @property
    def n_output(self):
        return self.config.output_size

    @property
    def n_sample(self):
        return self.batches.shape[0]

    def set_device_in_use(self, device):
        self.device_in_use = device

    def forward(
        self,
        modalities,
        batches,
        labels,
        cluster_requested=True,
        discriminator_requested=False,
    ):
        self.modalities = [
            modality.to(device=self.device_in_use) for modality in modalities
        ]
        self.batches = batches.to(device=self.device_in_use)
        self.labels = (
            labels.to(device=self.device_in_use) if labels is not None else None
        )

        self.latents = [
            encoder(modality)
            for (encoder, modality) in zip(self.encoders, self.modalities)
        ]

        self.translations = [
            [decoder(latent) for latent in self.latents] for decoder in self.decoders
        ]

        if discriminator_requested:
            self.discriminator_real_outputs = [
                discriminator(modality)
                for (discriminator, modality) in zip(
                    self.discriminators, self.modalities
                )
            ]

            self.discriminator_fake_outputs = [
                discriminator(self.translations[i][i].detach())
                for i, discriminator in enumerate(self.discriminators)
            ]

            self.generator_outputs = [
                discriminator(self.translations[i][i])
                for i, discriminator in enumerate(self.discriminators)
            ]

        if cluster_requested:
            self.fused_latents = [fuser(self.latents) for fuser in self.fusers]

            self.hiddens = [
                projector(fused_latent)
                for (projector, fused_latent) in zip(
                    self.projectors, self.fused_latents
                )
            ]

            self.cluster_outputs = [
                cluster(hidden)
                for (cluster, hidden) in zip(self.clusters, self.hiddens)
            ]

            return (
                self.translations,
                self.cluster_outputs[self.best_head],
                self.fused_latents[self.best_head],
            )

        else:
            return self.translations, None, None

    def save_model(self, path):
        """\
        Save current model parameters to the given path.

        path
            The absolute path to the desired location.
        """
        state_dict = {
            WEIGHTS: self.state_dict(),
            CONFIG: self.config,
            DATA_BATCH_ENCODER: self.data_batch_encoder.get_state(),
            DATA_LABEL_ENCODER: self.data_label_encoder.get_state(),
        }
        torch.save(state_dict, path)


# ==================== Model Generator ====================
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
    state_dict = torch.load(path)
    model = Model(state_dict[CONFIG])
    model.load_state_dict(state_dict[WEIGHTS])
    model.data_batch_encoder.set_state(state_dict[DATA_BATCH_ENCODER])
    model.data_label_encoder.set_state(state_dict[DATA_LABEL_ENCODER])
    return model
