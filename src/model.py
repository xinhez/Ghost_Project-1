import torch

from torch.nn import BatchNorm1d, Dropout, Linear
from torch.nn import ReLU, Sigmoid, Softmax, Tanh
from torch.nn import Module, ModuleList

from config import autocomplete_mlp_config, create_model_config_from_data
from config import validate_activation_method, validate_fusion_method
from constants import MEAN, RELU, SIGMOID, SOFTMAX, TANH
from utils import convert_to_lowercase


# ==================== Layer Generator ====================
def create_module_list(constructor, configs):
    """\
    Create a list of modules using the given constructor of the given configs.
    """
    return ModuleList([constructor(config) for config in configs])


def get_fuser(method):
    """\
    Get the fuser corresponding to the given fusion method.
    """
    method = convert_to_lowercase(method)
    validate_fusion_method(method)
    supported_fusion_methods = {
        MEAN: MeanFuser
    }
    return supported_fusion_methods[method]()


def get_activation(activation_method):
    """\
    Return the corresponding activation layer or raise exception if it is not supported.
    """
    validate_activation_method(activation_method)
    supported_activations = {
        RELU: ReLU(),
        SIGMOID: Sigmoid(),
        SOFTMAX: Softmax(dim=1),
        TANH: Tanh(),
    }
    return supported_activations[activation_method]


# ==================== Model Definition ====================
class MLP(Module):
    """
    MLP
    """
    def __init__(self, config):
        super().__init__()
        autocomplete_mlp_config(config)

        input_sizes = [config.input_size] + config.hidden_sizes
        output_sizes = config.hidden_sizes + [config.output_size]

        self.layers = ModuleList()

        for i in range(config.n_layers):
            self.layers.append(Linear(in_features=input_sizes[i], out_features=output_sizes[i], bias=config.use_biases[i]))

            if config.dropouts[i] > 0:
                self.layers.append(Dropout(p=config.dropouts[i]))

            if config.use_batch_norms[i]:
                self.layers.append(BatchNorm1d(output_sizes[i]))
            
            if config.activation_methods[i] is not None:
                self.layers.append(get_activation(config.activation_methods[i]))

    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MeanFuser(Module):
    """\
    Fuser
    """
    def __init__(self):
        super().__init__()
    

    def forward(self, x):
        return torch.mean(torch.stack(x, -1), dim=-1)


class Model(Module):
    """
    Model
    """
    def __init__(self, config):
        super().__init__()
        self.encoders       = create_module_list(MLP, config.encoders)
        self.decoders       = create_module_list(MLP, config.decoders)
        self.discriminators = create_module_list(MLP, config.discriminators)
        self.fuser          = get_fuser(config.fusion_method)
        self.cluster        = MLP(config.cluster)

        # self.optimizers
        pass


    def forward(self, modalities):
        self.modalities = modalities 

        self.latents = [
            encoder(modality) for (encoder, modality) in zip(self.encoders, modalities)
        ]

        self.translations = [
            [decoder(latent) for latent in self.latents] for decoder in self.decoders
        ]

        self.fused_latent = self.fuser(self.latents)

        self.cluster_output = self.cluster(self.fused_latent)

        self.discriminator_real_outputs = [
            discriminator(modality) 
            for (discriminator, modality) in zip(self.discriminators, modalities)
        ]

        self.discriminator_fake_outputs = [
            discriminator(self.translations[i][i]) 
            for i, discriminator in enumerate(self.discriminators)
        ]

        return self.fused_latent, self.translations, self.cluster_output


    def save_model(self, path):
        """\
        Save current model parameters to the given path.

        path
            The absolute path to the desired location.
        """
        torch.save(self, path)


# ==================== Model Generator ====================
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