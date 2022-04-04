from itertools import chain

from src.config import LossConfig
from src.model import ModuleNames
from src.managers.base import AlternativelyNamedObject, ObjectManager
from src.managers.loss import LossManager
from src.managers.loss import LatentMMDLoss, ReconstructionMMDLoss
from src.managers.loss import SCELoss
from src.managers.loss import SELoss, DDC1Loss, DDC3Loss
from src.managers.loss import ContrastiveLoss, DiscriminatorLoss, GeneratorLoss, ReconstructionLoss, TranslationLoss


class BaseSchedule(AlternativelyNamedObject):
    name = 'Schedule'


    def __init__(self, model, config):
        loss_configs = config.losses or self.loss_configs 
        self.losses = [
            LossManager.get_constructor_by_name(loss_config.name)(loss_config) 
            for loss_config in loss_configs
        ]

        optimizer_modules = config.optimizer_modules or self.optimizer_modules
        self.optimizers = list(chain.from_iterable([model.optimizers[module] for module in optimizer_modules]))


    def step(self, model):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        losses = {}
        for loss in self.losses:
            losses[loss.name] = loss(model)
            losses[loss.name].backward()

        for optimizer in self.optimizers:
            optimizer.step()

        return losses


class ClassificationSchedule(BaseSchedule):
    name = 'classification'
    loss_configs = [
        LossConfig(name=SCELoss.name), LossConfig(name=ReconstructionLoss.name), LossConfig(name=ContrastiveLoss.name)
    ]
    optimizer_modules = [
        ModuleNames.encoders, ModuleNames.fusers, ModuleNames.clusters
    ]


class ClusteringSchedule(BaseSchedule):
    name = 'clustering'
    loss_configs = [
        LossConfig(name=SELoss.name), LossConfig(name=DDC1Loss.name), LossConfig(name=DDC3Loss.name), LossConfig(name=ReconstructionLoss.name)
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
        LossConfig(name=ContrastiveLoss.name), 
        LossConfig(name=ReconstructionLoss.name), LossConfig(name=TranslationLoss.name), 
        LossConfig(name=DiscriminatorLoss.name),  LossConfig(name=GeneratorLoss.name),
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