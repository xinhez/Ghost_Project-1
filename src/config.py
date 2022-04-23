from pydantic import BaseModel
from typing import List, NewType, Union

import src.utils as utils

from src.managers.activation import ReLUActivation
from src.models.fuser import WeightedMeanFuser


# ==================== New Type Generators ====================
def none_or_type(t):
    """\
    Return new type defined as union of None or the given type
        e.g. bool -> Union[None, bool]
    """
    name = f"None_or_{t.__name__}"
    return NewType(name, Union[None, t])


def none_or_typelist(t):
    """\
    Return new type defined as union of None or list of the given type.
        e.g. bool -> Union[None, List[bool]]
    """
    name = f"None_or_{t.__name__}List"
    return NewType(name, Union[None, List[t]])


def none_or_type_or_typelist(t):
    """\
    Return new type defined as union of None, the given type or the list of the given type.
        e.g. bool -> Union[None, bool, List[bool]]
    """
    name = f"None_or_{t.__name__}_or_{t.__name__}List"
    return NewType(name, Union[None, t, List[t]])


def type_or_typelist(t):
    """\
    Return new type defined as the union of the given type or its list
        e.g. bool -> Union[bool, List[bool]]
            t: type to consider
    """
    name = f"{t.__name__}_or_{t.__name__}List"
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
    method: str = WeightedMeanFuser.name


# ==================== MLP Config Definition ====================
class MLPConfig(Config):
    # ===== sizes =====
    input_size: none_or_type(int) = None
    output_size: none_or_type(int) = None
    hidden_sizes: none_or_typelist(int) = None
    is_binary_input: none_or_type(bool) = False
    # ===== parameters =====
    activations: none_or_type_or_typelist(none_or_type(ActivationConfig)) = None
    dropouts: none_or_type_or_typelist(float) = None
    use_biases: none_or_type_or_typelist(bool) = None
    use_batch_norms: none_or_type_or_typelist(bool) = None
    use_layer_norms: none_or_type_or_typelist(bool) = None

    @property
    def n_layer(self):
        return 1 + len(self.hidden_sizes)


# ==================== Optimizer Config Definition ====================
class SchedulerConfig(Config):
    gamma: float = 0.1
    step_size: int = 50


class OptimizerConfig(Config):
    modules: none_or_typelist(str) = None
    clip_norm: none_or_type(float) = 25.0
    scheduler: none_or_type(SchedulerConfig) = None


# ==================== Loss Config Definition ====================
class LossConfig(Config):
    name: str
    weight: none_or_type(float) = None
    tau: none_or_type(float) = None
    sampling_ratio: none_or_type(float) = None
    sigmas: none_or_typelist(int) = None
    ref_batch: none_or_type(int) = None


# ==================== Schedule Config Definition ====================
class ScheduleConfig(Config):
    name: str
    best_loss_term: str = None
    losses: none_or_typelist(LossConfig) = None
    optimizer: OptimizerConfig = OptimizerConfig()


# ==================== Task Config Definition ====================
class TaskNames:
    cross_model_prediction = "cross_model_prediction"
    supervised_group_identification = "supervised_group_identification"
    unsupervised_group_identification = "unsupervised_group_identification"


# ==================== Technique Config Definition ====================
class TechniqueConfig(Config):
    latent_size: none_or_type(int) = None
    hidden_size: none_or_type(int) = None
    autoencoder_hidden_sizes: none_or_typelist(int) = None
    discriminator_hidden_sizes: none_or_typelist(int) = None
    autoencoder_use_batch_norms: none_or_type_or_typelist(bool) = None
    autoencoder_use_layer_norms: none_or_type_or_typelist(bool) = None
    n_head: none_or_type(int) = None
    fusion_method: none_or_type(str) = None


# ==================== Model Config Definition ====================
class ModuleNames:
    encoders = "encoders"
    decoders = "decoders"
    discriminators = "discriminators"
    fusers = "fusers"
    projectors = "projectors"
    clusters = "clusters"


class ModelConfig(Config):
    # ===== sizes =====
    input_sizes: List[int] = None
    output_size: none_or_type(int) = None
    n_batch: int = None
    class_weights: none_or_typelist(float) = None
    # ===== architecture =====
    encoders: List[MLPConfig] = None
    decoders: List[MLPConfig] = None
    discriminators: List[MLPConfig] = None
    fusers: List[FuserConfig] = None
    projectors: List[MLPConfig] = None
    clusters: List[MLPConfig] = None


def combine_mlpconfigs(current_config, new_config):
    if current_config is None:
        return new_config
    elif new_config is None:
        return current_config
    else:
        return MLPConfig(
            input_size=utils.get_new_or_current(
                current_config.input_size, new_config.input_size
            ),
            output_size=utils.get_new_or_current(
                current_config.output_size, new_config.output_size
            ),
            hidden_sizes=utils.get_new_or_current(
                current_config.hidden_sizes, new_config.hidden_sizes
            ),
            is_binary_input=utils.get_new_or_current(
                current_config.is_binary_input, new_config.is_binary_input
            ),
            activations=utils.get_new_or_current(
                current_config.activations, new_config.activations
            ),
            dropouts=utils.get_new_or_current(
                current_config.dropouts, new_config.dropouts
            ),
            use_biases=utils.get_new_or_current(
                current_config.use_biases, new_config.use_biases
            ),
            use_batch_norms=utils.get_new_or_current(
                current_config.use_batch_norms, new_config.use_batch_norms
            ),
            use_layer_norms=utils.get_new_or_current(
                current_config.use_layer_norms, new_config.use_layer_norms
            ),
        )


def combine_fuserconfigs(current_config, new_config):
    if current_config is None:
        return new_config
    elif new_config is None:
        return current_config
    else:
        return FuserConfig(
            n_modality=utils.get_new_or_current(
                current_config.n_modality, new_config.n_modality
            ),
            method=utils.get_new_or_current(current_config.method, new_config.method),
        )


def combine_config_lists(combiner, current_configs, new_configs):
    if current_configs is None:
        return new_configs
    elif new_configs is None:
        return current_configs
    else:
        combined_configs = []
        for current_config, new_config in zip(current_configs, new_configs):
            combined_configs.append(combiner(current_config, new_config))
        return combined_configs


def combine_configs(current_config, new_config):
    if current_config is None:
        return new_config
    elif new_config is None:
        return current_config
    else:
        return ModelConfig(
            input_sizes=current_config.input_sizes,
            output_size=current_config.output_size,
            n_batch=utils.get_new_or_current(
                current_config.n_batch, new_config.n_batch
            ),
            class_weights=current_config.class_weights,
            encoders=combine_config_lists(
                combine_mlpconfigs, current_config.encoders, new_config.encoders
            ),
            decoders=combine_config_lists(
                combine_mlpconfigs, current_config.decoders, new_config.decoders
            ),
            discriminators=combine_config_lists(
                combine_mlpconfigs,
                current_config.discriminators,
                new_config.discriminators,
            ),
            fusers=combine_config_lists(
                combine_fuserconfigs, current_config.fusers, new_config.fusers
            ),
            projectors=combine_config_lists(
                combine_mlpconfigs, current_config.projectors, new_config.projectors
            ),
            clusters=combine_config_lists(
                combine_mlpconfigs, current_config.clusters, new_config.clusters
            ),
        )
