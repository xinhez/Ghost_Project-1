from pydantic import BaseModel
from typing import List, NewType, Union

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


# ==================== Activation Config Definition ====================
class ActivationConfig(Config):
    method: str = ReLUActivation.name


# ==================== Fuser Config Definition ====================
class FuserConfig(Config):
    n_modality: int
    method:     str = WeightedMeanFuser.name


# ==================== MLP Config Definition ====================
class MLPConfig(Config):
    # ===== sizes =====
    input_size:          int
    output_size:         int
    hidden_sizes:        List[int]
    is_binary_input:     bool
    # ===== parameters =====
    activations:     type_or_typelist(none_or_type(ActivationConfig)) = None
    dropouts:        type_or_typelist(float)                          = 0
    use_biases:      type_or_typelist(bool)                           = False
    use_batch_norms: type_or_typelist(bool)                           = True


    @property
    def n_layer(self):
        return 1 + len(self.hidden_sizes)


# ==================== Optimizer Config Definition ====================
class SchedulerConfig(Config):
    gamma:     float = 0.1
    step_size: int   = 50


class OptimizerConfig(Config):
    learning_rate: float                         = 0.01
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
    losses:            none_or_typelist(LossConfig) = None
    optimizer_modules: none_or_typelist(str)        = None

    
# ==================== Model Config Definition ====================
class ModelConfig(Config):
    # ===== sizes =====
    input_sizes:   List[int]
    output_size:   none_or_type(int)
    n_batch:       int
    class_weights: none_or_typelist(float)
    # ===== architecture =====
    encoders:       List[MLPConfig]
    decoders:       List[MLPConfig]
    discriminators: List[MLPConfig]
    fusers:         List[FuserConfig]
    clusters:       List[MLPConfig]
    optimizers:     Optimizers


def combine_config(current_config, new_config):
    if current_config is None:
        return new_config
    else:
        return ModelConfig(
            input_sizes    = current_config.input_sizes,
            output_size    = current_config.output_size,
            n_batch        = current_config.n_batch,
            encoders       = new_config.encoders or current_config.encoders, 
            decoders       = new_config.decoders or current_config.decoders,
            discriminators = new_config.discriminators or current_config.new_config,
            fusers         = new_config.fusers or current_config.fusers,
            clusters       = new_config.clusters or current_config.clusters,
            optimizers     = new_config.optimizers or current_config.optimizers,
        )