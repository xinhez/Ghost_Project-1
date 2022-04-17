import numpy as np
import os
import tensorflow as tf
import torch

from src.config import LossConfig
from src.model import ModuleNames
from src.managers.base import AlternativelyNamedObject, ObjectManager
from src.managers.loss import LossManager
from src.managers.loss import LatentMMDLoss, ReconstructionMMDLoss
from src.managers.loss import CrossEntropyLoss
from src.managers.loss import SelfEntropyLoss, DDC1Loss, DDC3Loss
from src.managers.loss import (
    ContrastiveLoss,
    DiscriminatorLoss,
    GeneratorLoss,
    ReconstructionLoss,
    TranslationLoss,
)
from src.utils import sum_value_lists


class BaseSchedule(AlternativelyNamedObject):
    name = "Schedule"

    def __init__(self, logger, model, config, model_path, task, method, order):
        self.logger = logger

        loss_configs = self.loss_configs or config.losses
        if loss_configs is None:
            raise Exception(f"Please provide loss configs for {self.name} schedule.")
        self.losses = [
            LossManager.get_constructor_by_name(loss_config.name)(loss_config, model)
            for loss_config in loss_configs
        ]

        optimizer_modules = self.optimizer_modules or config.optimizer.modules
        if optimizer_modules is None:
            raise Exception(
                f"Please provide optimizer modules for {self.name} schedule."
            )
        model.create_optimizer_for_schedule(
            config.optimizer, self.name, optimizer_modules
        )

        self.order = order
        if model_path is None:
            self.model_path = None
        else:
            self.model_path = f"{model_path}/{task}_{method}_{self.order}_{self.name}"
            os.makedirs(self.model_path, exist_ok=True)

        self.cluster_requested = any([loss.based_on_head for loss in self.losses])
        self.discriminator_requested = any(
            [loss.based_on_discriminator for loss in self.losses]
        )

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

    def log_losses(self, writer, losses, epoch):
        with writer.as_default():
            for term in losses:
                tf.summary.scalar(
                    f"{self.model_path}/{term}",
                    losses[term].detach().cpu().numpy(),
                    step=epoch,
                )

    def save_model(self, model, name="best.pt"):
        if self.model_path is None:
            raise Exception("Please provide models saving path.")
        fullname = f"{self.model_path}/{name}"
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
                accumulated_head_losses = sum_value_lists(
                    accumulated_head_losses, head_losses
                )

        if train_model:
            # Step the optimizers
            total_loss.backward()
            model.optimizers_by_schedule[self.name].step()

            # Set the best head if the losses are based on head.
            if len(accumulated_head_losses) > 0:
                model.best_head = torch.argmin(torch.tensor(accumulated_head_losses))

        return losses


class CustomizedSchedule(BaseSchedule):
    name = "customized"
    loss_configs = None
    optimizer_modules = None


class ClassificationSchedule(BaseSchedule):
    name = "classification"
    loss_configs = [
        LossConfig(name=CrossEntropyLoss.name),
        LossConfig(name=ReconstructionLoss.name),
        LossConfig(name=ContrastiveLoss.name),
    ]
    optimizer_modules = [ModuleNames.encoders, ModuleNames.fusers, ModuleNames.clusters]


class ClassificationFinetuneSchedule(BaseSchedule):
    name = "classification_finetune"
    loss_configs = [
        LossConfig(name=CrossEntropyLoss.name),
    ]
    optimizer_modules = [ModuleNames.encoders, ModuleNames.fusers, ModuleNames.clusters]


class ClusteringSchedule(BaseSchedule):
    name = "clustering"
    loss_configs = [
        LossConfig(name=SelfEntropyLoss.name),
        LossConfig(name=DDC1Loss.name),
        LossConfig(name=DDC3Loss.name),
        LossConfig(name=ReconstructionLoss.name),
    ]
    optimizer_modules = [ModuleNames.fusers, ModuleNames.clusters]


class ClusteringFinetuneSchedule(BaseSchedule):
    name = "clustering_finetune"
    loss_configs = [
        LossConfig(name=SelfEntropyLoss.name),
        LossConfig(name=DDC1Loss.name),
        LossConfig(name=DDC3Loss.name),
        LossConfig(name=ReconstructionLoss.name),
    ]
    optimizer_modules = [ModuleNames.fusers, ModuleNames.clusters]


class TranslationSchedule(BaseSchedule):
    name = "translation"
    loss_configs = [
        LossConfig(name=ContrastiveLoss.name),
        LossConfig(name=ReconstructionLoss.name),
        LossConfig(name=TranslationLoss.name),
        LossConfig(name=DiscriminatorLoss.name),
        LossConfig(name=GeneratorLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.encoders,
        ModuleNames.decoders,
        ModuleNames.discriminators,
    ]


class TranslationFinetuneSchedule(BaseSchedule):
    name = "translation_finetune"
    loss_configs = [
        LossConfig(name=ReconstructionLoss.name),
        LossConfig(name=TranslationLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.encoders,
        ModuleNames.decoders,
        ModuleNames.discriminators,
    ]


class LatentBatchAlignmentSchedule(BaseSchedule):
    name = "latent_batch_alignment"
    loss_configs = [LossConfig(name=LatentMMDLoss.name)]
    optimizer_modules = [ModuleNames.encoders]


class ReconstructionBatchAlignmentSchedule(BaseSchedule):
    name = "reconstruction_batch_alignment"
    loss_configs = [LossConfig(name=ReconstructionMMDLoss.name)]
    optimizer_modules = [
        ModuleNames.encoders,
        ModuleNames.decoders,
        ModuleNames.discriminators,
    ]


class ScheduleManager(ObjectManager):
    """\
    Schedule

    Each schedule determines which losses to compute and which optimizers should step.
    """

    name = "schedules"
    constructors = [
        CustomizedSchedule,
        ClassificationSchedule,
        ClusteringSchedule,
        TranslationSchedule,
        LatentBatchAlignmentSchedule,
        ReconstructionBatchAlignmentSchedule,
    ]
