from pydantic import constr
from torch.nn import Module, ModuleList
from torch.nn import BatchNorm1d, Dropout, Linear

from src.managers.activation import ActivationManager, SoftmaxActivation
from src.utils import convert_to_lowercase


class MLP(Module):
    def __init__(self, config):
        super().__init__()
        autocomplete_mlp_config(config)

        input_sizes = [config.input_size] + config.hidden_sizes
        output_sizes = config.hidden_sizes + [config.output_size]

        self.layers = ModuleList()

        for i in range(config.n_layer):
            self.layers.append(Linear(in_features=input_sizes[i], out_features=output_sizes[i], bias=config.use_biases[i]))

            if config.dropouts[i] > 0:
                self.layers.append(Dropout(p=config.dropouts[i]))

            if config.use_batch_norms[i]:
                self.layers.append(BatchNorm1d(output_sizes[i]))
            
            if config.activation_methods[i] is not None:
                if (i+1) == config.n_layer and config.is_binary_input:
                    constructor = SoftmaxActivation
                else:
                    constructor = ActivationManager.get_constructor_by_name(config.activation_methods[i])
                if constructor == SoftmaxActivation:
                    self.layers.append(constructor(dim=1))
                else:
                    self.layers.append(constructor())

    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def autocomplete_mlp_config_attribute(config, attribute):
    """\
        Complete singleton attribute to a list if its type allows. 
        Convert all upper cases to lower cases during the process.
    """
    config_to_complete = getattr(config, attribute)
    if isinstance(config_to_complete, list):
        setattr(config, attribute, [convert_to_lowercase(c) for c in config_to_complete])
    else:
        config_to_complete = convert_to_lowercase(config_to_complete)
        setattr(config, attribute, [config_to_complete] * config.n_layer)


def autocomplete_mlp_config(config):
    """\
        Complete singleton config to a list if its type allows.
    """
    for attribute in config.attribute_of_variable_length:
        autocomplete_mlp_config_attribute(config, attribute)