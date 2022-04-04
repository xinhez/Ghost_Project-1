from pydantic import BaseModel
from typing import List, NewType, Union

from src.constants import mlp_config_attribute_of_variable_length
from src.managers.activation import ReLUActivation
from src.managers.fuser import WeightedMeanFuser


# ==================== New Type Generators ====================
def none_or_type(t):
    """\
    Return new type defined as union of None or the given type
        e.g. bool -> Union[None, bool]
    """
    name = f'None_or_{t.__name__}'
    return NewType(name, Union[None, t])


def none_or_typelist(t):
    """\
    Return new type defined as union of None or list of the given type.
        e.g. bool -> Union[None, List[bool]]
    """
    name = f'None_or_{t.__name__}List'
    return NewType(name, Union[None, List[t]])


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


# ==================== Fuser Config Definition ====================
class FuserConfig(Config):
    n_modality: int
    method:     str = 'weighted_mean'


# ==================== MLP Config Definition ====================
class MLPConfig(Config):
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
    attribute_of_variable_length = [*mlp_config_attribute_of_variable_length]


    @property
    def n_layer(self):
        return 1 + len(self.hidden_sizes)


def create_MLP_config(input_size, output_size, hidden_sizes):
    """\
    create_MLP_config
    """
    return MLPConfig(
        input_size   = input_size,
        output_size  = output_size,
        hidden_sizes = hidden_sizes,
    )


# ==================== Optimizer Config Definition ====================
class SchedulerConfig(Config):
    gamma:     float = 0.1
    step_size: int   = 50


class OptimizerConfig(Config):
    learning_rate: float                         = 0.001
    clip_norm:     none_or_type(float)           = 25.0
    scheduler:     none_or_type(SchedulerConfig) = None


class Optimizers(Config):
    encoders:       List[OptimizerConfig]
    decoders:       List[OptimizerConfig]
    discriminators: List[OptimizerConfig]
    fusers:         List[OptimizerConfig]
    clusters:       List[OptimizerConfig]


# ==================== Loss Config Definition ====================
class LossConfig(Config):
    name:   str
    weight: float = 1


# ==================== Schedule Config Definition ====================
class ScheduleConfig(Config):
    name:              str
    losses:            none_or_typelist(str) = None
    optimizer_modules: none_or_typelist(str) = None

    
# ==================== Model Config Definition ====================
class ModelConfig(Config):
    # ===== sizes =====
    input_sizes: List[int]
    output_size: int
    n_batch:     int
    # ===== architecture =====
    encoders:       List[MLPConfig]
    decoders:       List[MLPConfig]
    discriminators: List[MLPConfig]
    fusers:         List[FuserConfig]
    clusters:       List[MLPConfig]
    optimizers:     Optimizers

    @property
    def n_heads(self):
        if len(self.fusers) != len(self.clusters):
            raise Exception("Model must have the same number of fusers and clusters.")
        return len(self.fusers)


class Default():
    latent_size                = 16
    discriminator_output_size  = 32
    autoencoder_hidden_sizes   = [16, 16]
    discriminator_hidden_sizes = [64]
    n_head                     = 3
    fusion_method              = WeightedMeanFuser.name
    cluster_hidden_sizes       = [100]


def create_model_config(
    input_sizes, output_size, n_batch, 
    encoders_configs       = None,
    decoders_configs       = None,
    discriminators_configs = None, 
    fusers_configs         = None, 
    clusters_configs       = None,
    optimizers             = None,
):
    """\
    create_model_config
    """
    n_modality = len(input_sizes)

    return ModelConfig(
        input_sizes = input_sizes, 
        output_size = output_size,
        n_batch     = n_batch,
        encoders = encoders_configs or [
            create_MLP_config(input_size, Default.latent_size, Default.autoencoder_hidden_sizes) 
            for input_size in input_sizes
        ],
        decoders = decoders_configs or [
            create_MLP_config(Default.latent_size, input_size, list(reversed(Default.autoencoder_hidden_sizes)))
            for input_size in input_sizes
        ],
        discriminators = discriminators_configs or [
            create_MLP_config(input_size, Default.discriminator_output_size, Default.discriminator_hidden_sizes)
            for input_size in input_sizes
        ],
        fusers = fusers_configs or [
            FuserConfig(
                method=Default.fusion_method,
                n_modality=n_modality,
            ) 
            for _ in range(Default.n_head)
        ],
        clusters = clusters_configs or [
            create_MLP_config(Default.latent_size, output_size, Default.cluster_hidden_sizes) 
            for _ in range(Default.n_head)
        ],
        optimizers = optimizers or Optimizers(
            encoders       = [OptimizerConfig() for _ in input_sizes],
            decoders       = [OptimizerConfig() for _ in input_sizes],
            discriminators = [OptimizerConfig() for _ in input_sizes],
            fusers         = [OptimizerConfig() for _ in range(Default.n_head)],
            clusters       = [OptimizerConfig() for _ in range(Default.n_head)],
        )
    )


def create_model_config_from_data(data):
    """\
    Create a Model config with the sizes inferred from the given dataset.
    """
    return create_model_config(data.input_sizes, data.output_size, data.n_batch)