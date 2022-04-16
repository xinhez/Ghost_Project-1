import numpy as np
import os
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


    def __init__(self, logger, model, config, save_model_path, order):
        self.logger = logger

        loss_configs = config.losses or self.loss_configs 
        self.losses = [
            LossManager.get_constructor_by_name(loss_config.name)(loss_config, model)
            for loss_config in loss_configs
        ]

        model.create_optimizer_for_schedule(
            config.optimizer, self.name, config.optimizer.modules or self.optimizer_modules
        )

        self.order = order
        if save_model_path is None:
            self.save_model_path = None
        else:
            self.save_model_path = f'{save_model_path}/Schedule_{self.order}_{self.name}'
            os.makedirs(self.save_model_path, exist_ok=True)

        self.best_loss_term = config.best_loss_term
        self.best_loss = np.inf

    
    def check_and_update_best_loss(self, losses):
        if self.best_loss_term is None:
            loss = sum(losses.values())
        else:
            loss = losses[self.best_loss_term]
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        else:
            return False

    
    def save_model(self, model, name='best.pt'):
        if self.save_model_path is None:
            raise Exception("Please provide models saving path.")
        fullname = f'{self.save_model_path}/{name}'
        self.logger.log_save_model(fullname)
        model.save_model(fullname)


    def step(self, model, train_model):
        if train_model:
            model.optimizers_by_schedule[self.name].zero_grad()

        # Compute all losses
        losses = {}
        accumulated_head_losses = []
        total_loss = 0
        for loss in self.losses:
            losses[loss.name], head_losses = loss(model)
            total_loss += losses[loss.name]

            if train_model and head_losses is not None:
                accumulated_head_losses = sum_value_lists(accumulated_head_losses, head_losses)

        if train_model:
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
        LossConfig(name=ContrastiveLoss.name),
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
        LossConfig(name=ContrastiveLoss.name), 
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