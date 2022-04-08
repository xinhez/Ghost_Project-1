import torch

from src.config import LossConfig
from src.model import ModuleNames
from src.managers.base import AlternativelyNamedObject, ObjectManager
from src.managers.loss import LossManager
from src.managers.loss import LatentMMDLoss, ReconstructionMMDLoss
from src.managers.loss import CrossEntropyLoss
from src.managers.loss import SelfEntropyLoss, DDC1Loss, DDC3Loss
from src.managers.loss import ContrastiveLoss, DiscriminatorLoss, GeneratorLoss, ReconstructionLoss, TranslationLoss
from src.utils import sum_value_lists


class BaseSchedule(AlternativelyNamedObject):
    name = 'Schedule'


    def __init__(self, model, config):
        loss_configs = config.losses or self.loss_configs 
        self.losses = [
            LossManager.get_constructor_by_name(loss_config.name)(loss_config) 
            for loss_config in loss_configs
        ]

        model.create_optimizer_for_schedule(
            config.optimizer, self.name, config.optimizer.modules or self.optimizer_modules
        )


    def step(self, model):
        model.optimizers_by_schedule[self.name].zero_grad()


        # Compute all losses
        losses = {}
        accumulated_head_losses = []
        total_loss = 0
        for loss in self.losses:
            losses[loss.name], head_losses = loss(model)
            total_loss += losses[loss.name]

            if head_losses is not None:
                accumulated_head_losses = sum_value_lists(accumulated_head_losses, head_losses)

        # Step the optimizers
        total_loss.backward()
        model.optimizers_by_schedule[self.name].step()

        # Set the best head if the losses are based on head.
        if len(accumulated_head_losses) > 0:
            model.best_head = torch.argmin(torch.Tensor(accumulated_head_losses))

        return losses


class ClassificationSchedule(BaseSchedule):
    name = 'classification'
    loss_configs = [
        LossConfig(name=CrossEntropyLoss.name), LossConfig(name=ReconstructionLoss.name), 
        # LossConfig(name=ContrastiveLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.encoders, ModuleNames.fusers, ModuleNames.clusters
    ]


class ClusteringSchedule(BaseSchedule):
    name = 'clustering'
    loss_configs = [
        LossConfig(name=SelfEntropyLoss.name), 
        LossConfig(name=DDC1Loss.name), LossConfig(name=DDC3Loss.name), 
        LossConfig(name=ReconstructionLoss.name)
    ]
    optimizer_modules = [
        ModuleNames.fusers, ModuleNames.clusters
    ]


class LatentBatchAlignmentSchedule(BaseSchedule):
    name = 'latent_batch_alignment'
    loss_configs = [
        LossConfig(name=LatentMMDLoss.name)
    ]
    optimizer_modules = [
        ModuleNames.encoders
    ]


class TranslationSchedule(BaseSchedule):
    name = 'translation'
    loss_configs = [
        # LossConfig(name=ContrastiveLoss.name), 
        LossConfig(name=ReconstructionLoss.name), LossConfig(name=TranslationLoss.name), 
        # LossConfig(name=DiscriminatorLoss.name),  LossConfig(name=GeneratorLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.encoders, ModuleNames.decoders, ModuleNames.discriminators
    ]


class ReconstructionBatchAlignmentSchedule(TranslationSchedule):
    name = 'reconstruction_batch_alignment'
    loss_configs = [
        LossConfig(name=ReconstructionMMDLoss.name)
    ]
    optimizers = [
        ModuleNames.encoders, ModuleNames.decoders, ModuleNames.discriminators
    ]


class ScheduleManager(ObjectManager):
    """\
    Schedule

    Each schedule determines which losses to compute and which optimizers should step.
    """
    name = 'schedules'
    constructors = [
        ClassificationSchedule, ClusteringSchedule, TranslationSchedule,
        LatentBatchAlignmentSchedule, ReconstructionBatchAlignmentSchedule, 
    ]