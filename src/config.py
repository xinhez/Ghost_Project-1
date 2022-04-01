from pydantic import BaseModel
from typing import List, NewType, Union

from constants import mlp_attribute_of_variable_length
from managers.activation import ReLUActivation
from managers.fuser import MeanFuser


# ==================== New Type Generators ====================
def none_or_type(t):
    """\
    Return new type defined as union of None or the given type
        e.g. bool -> Union[None, bool]
    """
    name = f'None_or_{t.__name__}'
    return NewType(name, Union[None, t])


def type_or_typelist(t):
    """\
    Return new type defined as the union of the given type or its list
        e.g. bool -> Union[bool, List[bool]]
            t: type to consider
    """
    name = f'{t.__name__}_or_{t.__name__}List'
    return NewType(name, Union[t, List[t]])


# ==================== Config Definition ====================
class Config(BaseModel):
    @property
    def config_name(self):
        return self.__class__.__name__


class MLP(Config):
    # ===== sizes =====
    input_size:   int
    output_size:  int
    hidden_sizes: List[int]
    # ===== parameters =====
    dropouts:           type_or_typelist(float)             = 0
    use_biases:         type_or_typelist(bool)              = False
    use_batch_norms:    type_or_typelist(bool)              = True
    activation_methods: type_or_typelist(none_or_type(str)) = ReLUActivation.name
    # ===== static =====
    attribute_of_variable_length = [*mlp_attribute_of_variable_length]


    @property
    def n_layers(self):
        return 1 + len(self.hidden_sizes)

    
class Model(Config):
    # ===== sizes =====
    input_sizes: List[int]
    output_size: int
    n_batch:     int
    # ===== architecture =====
    encoders:       List[MLP]
    decoders:       List[MLP]
    discriminators: List[MLP]
    fusion_methods: List[str]
    clusters:       List[MLP]

    @property
    def n_heads(self):
        if len(self.fusion_method) != len(self.cluster):
            raise Exception("Model must have the same number of fusion methods and clusters.")
        return len(self.fusion_method)


def create_MLP_config(input_size, output_size, hidden_sizes):
    """\
    create_MLP_config
    """
    return MLP(
        input_size   = input_size,
        output_size  = output_size,
        hidden_sizes = hidden_sizes,
    )


# ==================== Config Creation ====================
def create_model_config(
    input_sizes, output_size, n_batch, 
    latent_size=16, discriminator_output_size=32,
    autoencoder_hidden_sizes=[16, 16], discriminator_hidden_sizes=[64],
    n_head = 3, fusion_method = MeanFuser.name, cluster_hidden_sizes=[100]
):
    """\
    create_model_config
    """
    return Model(
        input_sizes = input_sizes, 
        output_size = output_size,
        n_batch     = n_batch,
        encoders = [
            create_MLP_config(input_size, latent_size, autoencoder_hidden_sizes) 
            for input_size in input_sizes
        ],
        decoders = [
            create_MLP_config(latent_size, input_size, list(reversed(autoencoder_hidden_sizes)))
            for input_size in input_sizes
        ],
        discriminators = [
            create_MLP_config(input_size, discriminator_output_size, discriminator_hidden_sizes)
            for input_size in input_sizes
        ],
        fusion_methods = [fusion_method for _ in range(n_head)],
        clusters = [create_MLP_config(latent_size, output_size, cluster_hidden_sizes) for _ in range(n_head)],
    )


def create_model_config_from_data(data):
    """\
    Create a Model config with the sizes inferred from the given dataset.
    """
    return create_model_config(data.input_sizes, data.output_size, data.n_batches)