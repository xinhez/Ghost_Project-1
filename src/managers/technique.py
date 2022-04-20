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

    n_head = 1
    fusion_method = WeightedMeanFuser.name

    def get_default_config(self, data):
        n_autoencoder_layer = 1 + len(self.autoencoder_hidden_sizes)
        n_discriminator_layer = 1 + len(self.discriminator_hidden_sizes)

        return ModelConfig(
            input_sizes=data.modality_sizes,
            output_size=data.n_label,
            n_batch=data.n_batch,
            class_weights=data.class_weights,
            encoders=[
                MLPConfig(
                    input_size=input_size,
                    output_size=self.latent_size,
                    hidden_sizes=self.autoencoder_hidden_sizes,
                    is_binary_input=data.binary_modality_flags[modality_index],
                    activations=ActivationConfig(method=ReLUActivation.name),
                )
                for modality_index, input_size in enumerate(data.modality_sizes)
            ],
            decoders=[
                MLPConfig(
                    input_size=self.latent_size,
                    output_size=input_size,
                    hidden_sizes=list(reversed(self.autoencoder_hidden_sizes)),
                    activations=[
                        ActivationConfig(method=SigmoidActivation.name)
                        if n_layer + 1 == n_autoencoder_layer
                        and data.binary_modality_flags[modality_index]
                        else None
                        if (
                            n_layer + 1 == n_autoencoder_layer
                            and not data.positive_modality_flags[modality_index]
                        )
                        else ActivationConfig(method=ReLUActivation.name)
                        for n_layer in range(n_autoencoder_layer)
                    ],
                    use_batch_norms=[
                        n_layer + 1 != n_autoencoder_layer
                        for n_layer in range(n_autoencoder_layer)
                    ],
                )
                for modality_index, input_size in enumerate(data.modality_sizes)
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
                )
                for input_size in data.modality_sizes
            ],
            fusers=[
                FuserConfig(method=self.fusion_method, n_modality=data.n_modality,)
                for _ in range(self.n_head)
            ],
            projectors=[
                MLPConfig(input_size=self.latent_size, output_size=self.hidden_size,)
                for _ in range(self.n_head)
            ],
            clusters=[
                MLPConfig(input_size=self.hidden_size, output_size=data.n_label,)
                for _ in range(self.n_head)
            ],
        )


class ATACSeqTechnique(DefaultTechnique):
    name = "atacseq"
    

class DLPFCTechnique(DefaultTechnique):
    name = "dlpfc"
    
    latent_size = 50

    autoencoder_hidden_sizes = [256, 128, 64]


class TechniqueManager(ObjectManager):
    name = "techniques"
    constructors = [DefaultTechnique, ATACSeqTechnique, DLPFCTechnique]
