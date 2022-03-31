from pydantic import BaseModel
from typing import List, Literal, NewType, Union

from utils import convert_to_lowercase


# ==================== New Type Generators
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
    biases:         type_or_typelist(bool)              = False
    dropouts:       type_or_typelist(float)             = 0
    batch_norms:    type_or_typelist(bool)              = True
    activations:    type_or_typelist(none_or_type(str)) = 'ReLU'
    # ===== static =====
    attribute_of_variable_length = ['biases', 'dropouts', 'activations', 'batch_norms']


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
    fusion_method:  Literal['mean'] = 'mean'
    cluster:        MLP


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
    autoencoder_hidden_sizes=[16, 16], cluster_hidden_sizes=[100], discriminator_hidden_sizes=[64]
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
        cluster = create_MLP_config(latent_size, output_size, cluster_hidden_sizes),
        discriminators = [
            create_MLP_config(input_size, discriminator_output_size, discriminator_hidden_sizes)
            for input_size in input_sizes
        ]
    )


def create_model_config_from_data(data):
    """\
    Create a Model config with the sizes inferred from the given dataset.
    """
    return create_model_config(data.input_sizes, data.output_size, data.n_batch)


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
        setattr(config, attribute, [config_to_complete] * config.n_layers)


def autocomplete_mlp_config(config):
    """\
        Complete singleton config to a list if its type allows.
    """
    for attribute in config.attribute_of_variable_length:
        autocomplete_mlp_config_attribute(config, attribute)


# ==================== Config Validation ====================
def validate_config_layer_count(config, attribute):
    """\
    Check that the layers noted in the given configuration matches the given layer count.
    """
    config_to_check = getattr(config, attribute)
    if isinstance(config_to_check, list) and len(config_to_check) != config.n_layers:
        raise Exception(f"{attribute} in {config.config_name} does not match the {config.n_layers} layer count.")


def validate_mlp_config(config):
    """\
    Validate MLP config
    """
    for attribute in config.attribute_of_variable_length:
        validate_config_layer_count(config, attribute)