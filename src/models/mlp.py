import torch.nn as nn

from src.config import ActivationConfig
from src.managers.activation import ActivationManager, SoftmaxActivation


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_sizes = [config.input_size] + config.hidden_sizes
        output_sizes = config.hidden_sizes + [config.output_size]

        self.layers = nn.ModuleList()

        for i in range(config.n_layer):
            self.add_linear_layer(input_sizes[i], output_sizes[i], config.use_biases, i)
            self.add_dropout_layer(config.dropouts, i)
            self.add_batch_norms_layer(config.use_batch_norms, i, output_sizes[i])
            self.add_activation_layer (config.activations, i)


    def add_linear_layer(self, input_size, output_size, use_biases, i):
        use_bias = use_biases[i] if isinstance(use_biases, list) else use_biases
        self.layers.append(
            nn.Linear(in_features=input_size, out_features=output_size, bias=use_bias)
        )

    
    def add_dropout_layer(self, dropouts, i):
        if isinstance(dropouts, list) and dropouts[i] > 0:
            self.layers.append(nn.Dropout(p=dropouts[i]))
        elif isinstance(dropouts, float) and dropouts > 0:
            self.layers.append(nn.Dropout(p=dropouts))
    

    def add_batch_norms_layer(self, use_batch_norms, i, output_size):
        if ((isinstance(use_batch_norms, list) and use_batch_norms[i]) or 
            (isinstance(use_batch_norms, bool) and use_batch_norms)):
            self.layers.append(nn.BatchNorm1d(output_size))
        
    
    def add_activation_layer(self, activations, i):
        if isinstance(activations, list) and activations[i] is not None:
            constructor = ActivationManager.get_constructor_by_name(activations[i].method)
        elif isinstance(activations, ActivationConfig):
            constructor = ActivationManager.get_constructor_by_name(activations.method)
        else:
            constructor = None

        if constructor == SoftmaxActivation:
            self.layers.append(constructor(dim=1))
        elif constructor is not None:
            self.layers.append(constructor())

    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x