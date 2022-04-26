from src.configs.config import (
    ActivationConfig,
    FuserConfig,
    ScheduleConfig,
    MLPConfig,
    ModelConfig,
    TaskNames,
    combine_mlpconfigs,
)
from src.managers.activation import ReLUActivation, SigmoidActivation
from src.managers.base import NamedObject, ObjectManager
from src.models.fuser import WeightedMeanFuser
from src.managers.schedule import (
    ClassificationSchedule,
    ClusteringSchedule,
    TranslationSchedule,
)
from src.managers.schedule import (
    ClassificationFinetuneSchedule,
    TranslationFinetuneSchedule,
)
from src.managers.schedule import (
    ClassificationTransferSchedule,
    TranslationTransferSchedule,
)


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

    train_schedules = {
        TaskNames.cross_model_prediction: [
            ScheduleConfig(name=ClassificationSchedule.name),
            ScheduleConfig(name=TranslationSchedule.name),
        ],
        TaskNames.supervised_group_identification: [
            ScheduleConfig(name=TranslationSchedule.name),
            ScheduleConfig(name=ClassificationSchedule.name),
        ],
        TaskNames.unsupervised_group_identification: [
            ScheduleConfig(name=TranslationSchedule.name),
            ScheduleConfig(name=ClusteringSchedule.name),
        ],
    }

    finetune_schedules = {
        TaskNames.cross_model_prediction: [
            ScheduleConfig(name=ClassificationFinetuneSchedule.name),
            ScheduleConfig(name=TranslationFinetuneSchedule.name),
        ],
        TaskNames.supervised_group_identification: [
            ScheduleConfig(name=TranslationFinetuneSchedule.name),
            ScheduleConfig(name=ClassificationFinetuneSchedule.name),
        ],
    }

    transfer_schedules = {
        TaskNames.cross_model_prediction: [
            ScheduleConfig(name=ClassificationTransferSchedule.name),
            ScheduleConfig(name=TranslationTransferSchedule.name),
        ],
        TaskNames.supervised_group_identification: [
            ScheduleConfig(name=TranslationTransferSchedule.name),
            ScheduleConfig(name=ClassificationTransferSchedule.name),
        ],
    }

    def get_train_schedules(self, task):
        if task not in self.train_schedules:
            raise Exception(
                f"Only {self.train_schedules.keys()} tasks are supported for {self.name} technique name for training."
            )
        return self.train_schedules[task]

    def get_finetune_schedules(self, task):
        if task not in self.train_schedules:
            raise Exception(
                f"Only {self.finetune_schedules.keys()} tasks are supported for {self.name} technique name for finetuning."
            )
        return self.finetune_schedules[task]

    def get_transfer_schedules(self, task):
        if task not in self.train_schedules:
            raise Exception(
                f"Only {self.transfer_schedules.keys()} tasks are supported for {self.name} technique name for transfering."
            )
        return self.transfer_schedules[task]

    def __init__(self, data):
        self.modality_sizes = data.modality_sizes
        self.n_label = data.n_label
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
            class_weights=self.class_weights,
            encoders=[
                MLPConfig(
                    input_size=input_size,
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
                    input_size=self.latent_size,
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


class PatchSeqTechnique(DefaultTechnique):
    name = "patchseq"

    latent_size = 68
    autoencoder_hidden_sizes = [1024, 512, 256, 128]
    autoencoder_autoencoder_use_layer_norms = [False, False, False, False, True]
    n_head = 10
    use_biases = True

    def get_model_config(self):
        config = super().get_model_config()

        n_autoencoder_layer = len(self.autoencoder_hidden_sizes) + 1

        for i, (encoder, decoder) in enumerate(zip(config.encoders, config.decoders)):
            config.encoders[i] = combine_mlpconfigs(
                encoder,
                MLPConfig(
                    dropouts=0 if i == 0 else [0.1, 0, 0, 0, 0],
                    activations=[
                        ActivationConfig(method=ReLUActivation.name)
                        if n_layer + 1 != n_autoencoder_layer
                        else None
                        for n_layer in range(n_autoencoder_layer)
                    ],
                    use_layer_norms=self.autoencoder_autoencoder_use_layer_norms,
                ),
            )

        for i, (projector, cluster) in enumerate(
            zip(config.projectors, config.clusters)
        ):
            config.projectors[i] = combine_mlpconfigs(
                projector,
                MLPConfig(
                    activations=ActivationConfig(method=ReLUActivation.name),
                    use_layer_norms=True,
                    use_biases=True,
                ),
            )
            config.clusters[i] = combine_mlpconfigs(
                cluster,
                MLPConfig(
                    use_biases=True,
                ),
            )

        return config


class DLPFCTechnique(DefaultTechnique):
    name = "dlpfc"

    latent_size = 50

    autoencoder_hidden_sizes = [256, 128, 64]
    discriminator_hidden_sizes = [64, 32]

    autoencoder_use_batch_norms = True

    def get_train_schedules(self, task):
        if task == TaskNames.supervised_group_identification:
            return [
                ScheduleConfig(name=TranslationSchedule.name),
                *[ScheduleConfig(name=ClassificationSchedule.name)] * self.n_modality,
            ]
        else:
            return super().get_train_schedules(task)

    def get_finetune_schedules(self, task):
        if task == TaskNames.supervised_group_identification:
            return [
                ScheduleConfig(name=TranslationSchedule.name),
                *[ScheduleConfig(name=ClassificationSchedule.name)] * self.n_modality,
            ]
        else:
            return super().get_finetune_schedules(task)

    def get_transfer_schedules(self, task):
        if task == TaskNames.supervised_group_identification:
            return [
                ScheduleConfig(name=TranslationSchedule.name),
                *[ScheduleConfig(name=ClassificationSchedule.name)] * self.n_modality,
            ]
        else:
            return super().get_transfer_schedules(task)

    def get_model_config(self):
        config = super().get_model_config()

        n_autoencoder_layer = len(self.autoencoder_hidden_sizes) + 1

        for i, encoder in enumerate(config.encoders):
            config.encoders[i] = combine_mlpconfigs(
                encoder,
                MLPConfig(
                    activations=[
                        ActivationConfig(method=ReLUActivation.name)
                        if n_layer + 1 != n_autoencoder_layer
                        else None
                        for n_layer in range(n_autoencoder_layer)
                    ]
                ),
            )

        return config


class TechniqueManager(ObjectManager):
    name = "techniques"
    constructors = [
        DefaultTechnique,
        ATACSeqTechnique,
        DLPFCTechnique,
        PatchSeqTechnique,
    ]
