import torch
from torch.nn import BatchNorm1d, Linear
from torch.nn import ReLU, Sigmoid, Softmax, Tanh
from torch.nn import Module, ModuleList

from data import validate_anndatas
from config import autocomplete_mlp_config, create_model_config_from_data, validate_mlp_config


def create_module_list(constructor, configs):
    """\
        Create a list of modules using the given constructor of the given configs.
    """
    return ModuleList([constructor(config) for config in configs])


def get_activation_layer(activation_type):
    """\
        Return the corresponding activation layer or raise exception if it is not supported.
    """
    supported_activations = {
        'relu': ReLU(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'softmax': Softmax(dim=1)
    }
    if activation_type.lower() in supported_activations.keys():
        return supported_activations[activation_type]


class MLP(Module):
    """
    MLP
    """
    def __init__(self, config):
        super().__init__()
        autocomplete_mlp_config(config)
        validate_mlp_config(config)

        input_sizes = [config.input_size] + config.hidden_sizes
        output_sizes = config.hidden_sizes + [config.output_size]

        self.layers = ModuleList()

        for i in range(config.n_layers):
            self.layers.append(Linear(in_features=input_sizes[i], out_features=output_sizes[i], bias=config.biases[i]))

            if config.dropouts[i] > 0:
                self.layers.append(Dropout(p=config.dropouts[i]))

            if config.batch_norms[i]:
                self.layers.append(BatchNorm1d(output_sizes[i]))
            
            if config.activations[i] is not None:
                self.layers.append(get_activation_layer(config.activations[i]))


class Model(Module):
    """
    Model
    """
    def __init__(self, config):
        super().__init__()
        self.encoders = create_module_list(MLP, config.encoders)
        self.decoders = create_module_list(MLP, config.decoders)
        self.discriminators = create_module_list(MLP, config.discriminators)
        # self.fusers
        # self.clusters
        # self.optimizers
        pass


    def save_model(self, path):
        """\
        Save current model parameters to the given path.

        path
            The absolute path to the desired location.
        """
        torch.save(self, path)


def create_model_from_data(
    adatas, 
    reference_index_label, reference_index_batch, 
    obs_key_label, obs_key_batch
):
    """\
    Create a new model based on the given dataset.
    """
    validate_anndatas(adatas, obs_key_label, reference_index_label)

    return Model(create_model_config_from_data(
        adatas, 
        reference_index_label, reference_index_batch, 
        obs_key_label, obs_key_batch
    ))


def load_model_from_path(path):
    """\
    Create a new model with model parameters from the given path.

    path
        The absolute path to the desired location.
    """
    return torch.load(path)