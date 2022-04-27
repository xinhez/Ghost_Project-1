from src.config import ActivationConfig, FuserConfig, MLPConfig, ModelConfig
from src.managers.activation import ReLUActivation, SigmoidActivation
from src.managers.base import NamedObject, ObjectManager
from src.models.fuser import WeightedMeanFuser


class DefaultTechnique(NamedObject):
    name = "default"

    latent_size = 64
    hidden_size = 100

    autoencoder_hidden_sizes = [64, 64]
    discriminator_hidden_sizes = [64]

    autoencoder_use_batch_norms = False
    autoencoder_use_layer_norms = False

    dropouts = 0
    use_biases = False

    n_head = 1
    fusion_method = WeightedMeanFuser.name

    def __init__(self, data):
        self.modality_sizes = data.modality_sizes
        self.n_label = data.n_label
        self.n_batch = data.n_batch
        self.n_modality = data.n_modality
        self.class_weights = data.class_weights
        self.binary_modality_flags = data.binary_modality_flags
        self.positive_modality_flags = data.positive_modality_flags

    def update_config(self, config):
        self.latent_size = config.latent_size or self.latent_size
        self.hidden_size = config.hidden_size or self.hidden_size
        self.autoencoder_hidden_sizes = (
            config.autoencoder_hidden_sizes or self.autoencoder_hidden_sizes
        )
        self.discriminator_hidden_sizes = (
            config.discriminator_hidden_sizes or self.discriminator_hidden_sizes
        )
        self.autoencoder_use_batch_norms = (
            config.autoencoder_use_batch_norms or self.autoencoder_use_batch_norms
        )
        self.autoencoder_use_layer_norms = (
            config.autoencoder_use_layer_norms or self.autoencoder_use_layer_norms
        )
        self.n_head = config.n_head or self.n_head
        self.fusion_method = config.fusion_method or self.fusion_method

    def get_model_config(self):
        n_autoencoder_layer = 1 + len(self.autoencoder_hidden_sizes)
        n_discriminator_layer = 1 + len(self.discriminator_hidden_sizes)

        return ModelConfig(
            input_sizes=self.modality_sizes,
            output_size=self.n_label,
            n_batch=self.n_batch,
            class_weights=self.class_weights,
            encoders=[
                MLPConfig(
                    input_size=input_size+self.n_batch,
                    output_size=self.latent_size,
                    hidden_sizes=self.autoencoder_hidden_sizes,
                    dropouts=self.dropouts,
                    use_biases=self.use_biases,
                    is_binary_input=self.binary_modality_flags[modality_index],
                    activations=ActivationConfig(method=ReLUActivation.name),
                    use_batch_norms=self.autoencoder_use_batch_norms,
                    use_layer_norms=self.autoencoder_use_layer_norms,
                )
                for modality_index, input_size in enumerate(self.modality_sizes)
            ],
            decoders=[
                MLPConfig(
                    input_size=self.latent_size+self.n_batch,
                    output_size=input_size,
                    hidden_sizes=list(reversed(self.autoencoder_hidden_sizes)),
                    dropouts=self.dropouts,
                    use_biases=self.use_biases,
                    is_binary_input=False,
                    activations=[
                        ActivationConfig(method=SigmoidActivation.name)
                        if n_layer + 1 == n_autoencoder_layer
                        and self.binary_modality_flags[modality_index]
                        else None
                        if (
                            n_layer + 1 == n_autoencoder_layer
                            and not self.positive_modality_flags[modality_index]
                        )
                        else ActivationConfig(method=ReLUActivation.name)
                        for n_layer in range(n_autoencoder_layer)
                    ],
                    use_batch_norms=[
                        (n_layer + 1 != n_autoencoder_layer)
                        and self.autoencoder_use_batch_norms
                        for n_layer in range(n_autoencoder_layer)
                    ],
                    use_layer_norms=[
                        (n_layer + 1 != n_autoencoder_layer)
                        and self.autoencoder_use_layer_norms
                        for n_layer in range(n_autoencoder_layer)
                    ],
                )
                for modality_index, input_size in enumerate(self.modality_sizes)
            ],
            discriminators=[
                MLPConfig(
                    input_size=input_size,
                    output_size=1,
                    hidden_sizes=self.discriminator_hidden_sizes,
                    activations=[
                        ActivationConfig(method=SigmoidActivation.name)
                        if n_layer + 1 == n_discriminator_layer
                        else ActivationConfig(method=ReLUActivation.name)
                        for n_layer in range(n_discriminator_layer)
                    ],
                    dropouts=self.dropouts,
                    use_biases=self.use_biases,
                    is_binary_input=False,
                    use_batch_norms=False,
                    use_layer_norms=False,
                )
                for input_size in self.modality_sizes
            ],
            fusers=[
                FuserConfig(method=self.fusion_method, n_modality=self.n_modality)
                for _ in range(self.n_head)
            ],
            projectors=[
                MLPConfig(
                    input_size=self.latent_size, 
                    output_size=self.hidden_size,
                    hidden_sizes=[],
                    dropouts=self.dropouts,
                    use_biases=self.use_biases,
                    is_binary_input=False,
                    activations=None,
                    use_batch_norms=False,
                    use_layer_norms=False,
                    )
                for _ in range(self.n_head)
            ],
            clusters=[
                MLPConfig(
                    input_size=self.hidden_size, 
                    output_size=self.n_label,
                    hidden_sizes=[],
                    dropouts=self.dropouts,
                    use_biases=self.use_biases,
                    is_binary_input=False,
                    activations=None,
                    use_batch_norms=False,
                    use_layer_norms=False,
                )
                for _ in range(self.n_head)
            ],
        )


class ATACSeqTechnique(DefaultTechnique):
    name = "atacseq"

    autoencoder_use_batch_norms = True


class DLPFCTechnique(DefaultTechnique):
    name = "dlpfc"

    latent_size = 50

    autoencoder_hidden_sizes = [256, 128, 64]
    discriminator_hidden_sizes = [64, 32]

    autoencoder_use_batch_norms = True


class TechniqueManager(ObjectManager):
    name = "techniques"
    constructors = [DefaultTechnique, ATACSeqTechnique, DLPFCTechnique]
