import numpy as np
import os
import tensorflow as tf
import torch

from src.config import LossConfig, ModuleNames
from src.managers.base import AlternativelyNamedObject, ObjectManager
from src.managers.loss import LossManager
from src.managers.loss import ReconstructionMMDLoss
from src.managers.loss import CrossEntropyLoss
from src.managers.loss import SelfEntropyLoss, DDCLoss
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
    best_loss_term = None
    loss_configs = None
    optimizer_modules = None

    def __init__(
        self, logger, model, learning_rate, config, model_path, method, order
    ):
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
            learning_rate, config.optimizer, self.name, optimizer_modules
        )

        self.order = order
        if model_path is None:
            self.model_path = None
        else:
            self.model_path = f"{model_path}/{method}_{self.order}_{self.name}"
            os.makedirs(self.model_path, exist_ok=True)

        self.cluster_requested = any([loss.based_on_head for loss in self.losses])
        self.discriminator_requested = any(
            [loss.based_on_discriminator for loss in self.losses]
        )

        self.best_loss_term = self.best_loss_term or config.best_loss_term
        self.best_loss = np.inf

    def check_and_update_best_loss(self, losses):
        if self.best_loss_term is None:
            loss = sum(losses.values())
        else:
            loss = losses[self.best_loss_term]
        if 0 < loss < self.best_loss:
            self.best_loss = loss
            return True
        else:
            return False

    def log_losses(self, writer, losses, epoch):
        if writer is not None:
            with writer.as_default():
                for term in losses:
                    tf.summary.scalar(
                        f"{self.model_path}/{term}",
                        losses[term].detach().cpu().numpy() if torch.is_tensor(losses[term]) else losses[term],
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


class ClassificationSchedule(BaseSchedule):
    name = "classification"
    best_loss_term = CrossEntropyLoss.name
    loss_configs = [
        LossConfig(name=CrossEntropyLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.encoders,
        ModuleNames.fusers,
        ModuleNames.projectors,
        ModuleNames.clusters,
    ]


class ClassificationFinetuneSchedule(ClassificationSchedule):
    name = "classification(finetune)"


class ClassificationTransferSchedule(ClassificationSchedule):
    name = "classification(transfer)"


class ClusteringSchedule(BaseSchedule):
    name = "clustering"
    loss_configs = [
        LossConfig(name=SelfEntropyLoss.name),
        LossConfig(name=DDCLoss.name),
        LossConfig(name=ReconstructionLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.fusers,
        ModuleNames.projectors,
        ModuleNames.clusters,
    ]


class ClusteringFinetuneSchedule(ClusteringSchedule):
    name = "clustering(finetune)"
    loss_configs = [
        LossConfig(name=SelfEntropyLoss.name),
        LossConfig(name=DDCLoss.name),
        LossConfig(name=ReconstructionLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.fusers,
        ModuleNames.projectors,
        ModuleNames.clusters,
    ]


class ClusteringTransferSchedule(ClusteringSchedule):
    name = "clustering(transfer)"
    loss_configs = [
        LossConfig(name=SelfEntropyLoss.name),
        LossConfig(name=DDCLoss.name),
        LossConfig(name=ReconstructionLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.fusers,
        ModuleNames.projectors,
        ModuleNames.clusters,
    ]


class TranslationSchedule(BaseSchedule):
    name = "translation"
    best_loss_term = TranslationLoss.name
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


class TranslationFinetuneSchedule(TranslationSchedule):
    name = "translation(finetune)"
    loss_configs = [
        LossConfig(name=ReconstructionLoss.name),
        LossConfig(name=TranslationLoss.name),
    ]


class TranslationTransferSchedule(TranslationSchedule):
    name = "translation(transfer)"
    loss_configs = [
        LossConfig(name=ReconstructionLoss.name),
        LossConfig(name=TranslationLoss.name),
        LossConfig(name=ContrastiveLoss.name),
    ]


class ReconstructionBatchAlignmentSchedule(BaseSchedule):
    name = "reconstruction_batch_alignment"
    best_loss_term = ReconstructionMMDLoss.name
    loss_configs = [
        LossConfig(name=ReconstructionMMDLoss.name),
        LossConfig(name=ReconstructionLoss.name),
        LossConfig(name=TranslationLoss.name),
    ]
    optimizer_modules = [
        ModuleNames.encoders,
        ModuleNames.decoders,
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
        ClassificationFinetuneSchedule,
        ClassificationTransferSchedule,
        ClusteringSchedule,
        ClusteringFinetuneSchedule,
        ClusteringTransferSchedule,
        TranslationSchedule,
        TranslationFinetuneSchedule,
        TranslationTransferSchedule,
        ReconstructionBatchAlignmentSchedule,
    ]
