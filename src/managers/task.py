import torch

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, r2_score
from src.managers.data import DataManager

import src.utils as utils

from src.config import ScheduleConfig
from src.managers.base import AlternativelyNamedObject, ObjectManager
from src.managers.schedule import ScheduleManager
from src.managers.schedule import (
    ClassificationSchedule,
    ClusteringSchedule,
    TranslationSchedule,
)
from src.managers.schedule import (
    ClassificationFinetuneSchedule,
    ClusteringFinetuneSchedule,
    TranslationFinetuneSchedule,
)
from src.managers.schedule import (
    LatentBatchAlignmentSchedule,
    ReconstructionBatchAlignmentSchedule,
)


class BaseTask(AlternativelyNamedObject):
    name = "Task"

    def update_schedules(self, logger, model, schedule_configs, model_path, method):
        if schedule_configs is None:
            raise Exception(f"Please provide {method}_schedules for {self.name} task.")
        self.schedules = [
            ScheduleManager.get_constructor_by_name(config.name)(
                logger,
                model,
                config,
                model_path,
                self.get_short_name(),
                method,
                schedule_order,
            )
            for (schedule_order, config) in enumerate(schedule_configs)
        ]

    def run_through_data(
        self,
        logger,
        model,
        dataloader,
        epoch=None,
        schedule=None,
        train_model=False,
        infer_model=False,
        save_best_model=False,
        checkpoint_model_name=None,
        writer=None,
    ):
        """\
        Losses can only computed by the schedule.
        Inference involves moving tensors to CPU and can be slow.
        """
        if train_model and schedule is None:
            raise Exception("Please provide training schedules.")

        all_outputs = []
        all_losses = {}

        for modalities, *batches_and_maybe_labels in dataloader:
            if len(batches_and_maybe_labels) == 1:
                batches, labels = *batches_and_maybe_labels, None
            elif len(batches_and_maybe_labels) == 2:
                batches, labels = batches_and_maybe_labels

            if schedule is not None:
                outputs = model(
                    modalities,
                    batches,
                    labels,
                    schedule.cluster_requested,
                    schedule.discriminator_requested,
                )
                losses = schedule.step(model, train_model)
                losses = utils.amplify_value_dictionary_by_sample_size(
                    losses, len(batches)
                )
                all_losses = utils.sum_value_dictionaries(all_losses, losses)
            else:
                outputs = model(modalities, batches, labels)

            if infer_model:
                outputs = utils.move_tensor_list_to_cpu(outputs)
                all_outputs = utils.combine_tensor_lists(all_outputs, outputs)

        if all_losses:
            all_losses = utils.average_dictionary_values_by_sample_size(
                all_losses, len(dataloader.dataset)
            )
            logger.log_losses(all_losses)
            if epoch is not None:
                schedule.log_losses(writer, all_losses, epoch)

        if save_best_model and schedule.check_and_update_best_loss(all_losses):
            schedule.save_model(model)

        if checkpoint_model_name is not None:
            schedule.save_model(model, checkpoint_model_name)

        return all_outputs

    def evaluate_outputs(self, logger, dataset, outputs):
        labels = dataset.labels
        translations_outputs, cluster_outputs, *_ = outputs
        predictions = cluster_outputs.argmax(axis=1)

        r2s = [
            [
                r2_score(modality.numpy(), translation.numpy())
                for translation in translations
            ]
            for modality, translations in zip(dataset.modalities, translations_outputs)
        ]
        accuracy = torch.sum(labels == predictions).data / predictions.shape[0]

        labels = labels.numpy()
        predictions = predictions.numpy()
        metrics = {
            "r2": r2s,
            "acc": accuracy,
            "ari": adjusted_rand_score(labels, predictions),
            "nmi": normalized_mutual_info_score(
                labels, predictions, average_method="geometric"
            ),
        }

        logger.log_metrics(metrics)
        return metrics

    def evaluate(self, model, data_eval, logger):
        logger.log_method_start(self.evaluate.__name__)

        dataloader_eval = data_eval.create_dataloader(model, shuffle=False)

        model.eval()
        with torch.no_grad():
            outputs = self.run_through_data(
                logger, model, dataloader_eval, infer_model=True
            )
        metrics = self.evaluate_outputs(logger, dataloader_eval.dataset, outputs)

        return metrics

    def infer(self, model, data_infer, logger, modalities_provided):
        logger.log_method_start(self.infer.__name__)

        dataloader_infer = data_infer.create_dataloader(model, shuffle=False)

        model.eval()

        with torch.no_grad():
            outputs = self.run_through_data(
                logger, model, dataloader_infer, infer_model=True
            )

        if len(modalities_provided) == 0 or data_infer.n_modality == len(
            modalities_provided
        ):
            return DataManager.anndata_from_outputs(
                model, dataloader_infer.dataset, outputs
            )

        raise Exception("Fill in missing modalities not implemented!")

    def train_with_schedules(
        self,
        logger,
        model,
        n_epoch,
        data,
        data_validate,
        batch_size,
        save_best_model,
        checkpoint,
        writer,
        random_seed,
    ):
        dataloader = data.create_dataloader(model, shuffle=True, batch_size=batch_size, random_seed=random_seed)
        dataloader_validate = data_validate.create_dataloader(
            model, shuffle=True, batch_size=batch_size, random_seed=random_seed
        )

        model.train()
        for epoch in range(n_epoch):
            epoch += 1
            logger.log_epoch_start(epoch, n_epoch)

            for schedule in self.schedules:
                logger.log_schedule_start(schedule)

                self.run_through_data(
                    logger,
                    model,
                    dataloader,
                    epoch,
                    schedule,
                    train_model=True,
                    writer=writer,
                )

                checkpoint_model_name = None
                if checkpoint > 0:
                    if epoch % checkpoint == 0:
                        checkpoint_model_name = f"epoch_{epoch}.pt"

                with torch.no_grad():
                    self.run_through_data(
                        logger,
                        model,
                        dataloader_validate,
                        schedule=schedule,
                        save_best_model=save_best_model,
                        checkpoint_model_name=checkpoint_model_name,
                    )

                writer.flush()

    def train(
        self,
        config,
        model,
        data,
        data_validate,
        batch_size,
        n_epoch,
        logger,
        model_path,
        save_best_model,
        checkpoint,
        writer,
        random_seed,
    ):
        logger.log_method_start(self.train.__name__, self.name)

        self.update_schedules(
            logger,
            model,
            self.train_schedule_configs or config.train_schedules,
            model_path,
            self.train.__name__,
        )

        self.train_with_schedules(
            logger,
            model,
            n_epoch,
            data,
            data_validate,
            batch_size,
            save_best_model,
            checkpoint,
            writer,
            random_seed,
        )

    def finetune(
        self,
        config,
        model,
        data,
        data_validate,
        batch_size,
        n_epoch,
        logger,
        model_path,
        save_best_model,
        checkpoint,
        writer,
        random_seed,
    ):
        logger.log_method_start(self.finetune.__name__, self.name)

        self.update_schedules(
            logger,
            model,
            self.finetune_schedule_configs or config.finetune_schedules,
            model_path,
            self.finetune.__name__,
        )

        self.train_with_schedules(
            logger,
            model,
            n_epoch,
            data,
            data_validate,
            batch_size,
            save_best_model,
            checkpoint,
            writer,
            random_seed,
        )

    def transfer(
        self,
        config,
        model,
        data,
        data_transfer,
        data_validate,
        batch_size,
        n_epoch,
        logger,
        model_path,
        save_best_model,
        checkpoint,
        writer,
        random_seed,
    ):
        logger.log_method_start(self.transfer.__name__, self.name)

        self.update_schedules(
            logger,
            model,
            self.transfer_schedule_configs or config.transfer_schedules,
            model_path,
            self.transfer.__name__,
        )

        dataloader = data.create_dataloader(model, shuffle=True, batch_size=batch_size, random_seed=random_seed)
        dataloader_train_and_transfer = data.create_joint_dataloader(
            data_transfer, model, shuffle=True, batch_size=batch_size
        )
        datalodaer_validate = data_validate.create_dataloader(
            model, shuffle=True, batch_size=batch_size, random_seed=random_seed,
        )

        for epoch in range(n_epoch):
            epoch += 1
            logger.log_epoch_start(epoch, n_epoch)
            for schedule in self.schedules:
                logger.log_schedule_start(schedule)
                if isinstance(schedule, ClassificationSchedule):
                    self.run_through_data(
                        logger,
                        model,
                        dataloader,
                        epoch,
                        schedule,
                        train_model=True,
                        writer=writer,
                    )
                else:
                    self.run_through_data(
                        logger,
                        model,
                        dataloader_train_and_transfer,
                        epoch,
                        schedule,
                        train_model=True,
                        writer=writer,
                    )

                checkpoint_model_name = None
                if checkpoint > 0:
                    if epoch % checkpoint == 0:
                        checkpoint_model_name = f"epoch_{epoch}.pt"

                with torch.no_grad():
                    self.run_through_data(
                        logger,
                        model,
                        datalodaer_validate,
                        schedule=schedule,
                        save_best_model=save_best_model,
                        checkpoint_model_name=checkpoint_model_name,
                    )

                writer.flush()


class CustomizedTask(BaseTask):
    name = "customized"
    train_schedule_configs = None
    finetune_schedule_configs = None
    transfer_schedule_configs = None


class CrossModelPredictionTask(BaseTask):
    name = "cross_model_prediction"
    train_schedule_configs = [
        ScheduleConfig(name=ClassificationSchedule.name),
        ScheduleConfig(name=TranslationSchedule.name),
    ]

    finetune_schedule_configs = [
        ScheduleConfig(name=ClassificationFinetuneSchedule.name),
        ScheduleConfig(name=TranslationFinetuneSchedule.name),
    ]

    transfer_schedule_configs = [
        ScheduleConfig(name=LatentBatchAlignmentSchedule.name),
        ScheduleConfig(name=ReconstructionBatchAlignmentSchedule.name),
        ScheduleConfig(name=ClassificationSchedule.name),
        ScheduleConfig(name=TranslationSchedule.name),
    ]


class SupervisedGroupIdentificationTask(BaseTask):
    name = "supervised_group_identification"
    train_schedule_configs = [
        ScheduleConfig(name=TranslationSchedule.name),
        ScheduleConfig(name=ClassificationSchedule.name),
    ]

    finetune_schedule_configs = [
        ScheduleConfig(name=TranslationFinetuneSchedule.name),
        ScheduleConfig(name=ClassificationFinetuneSchedule.name),
    ]

    transfer_schedule_configs = [
        ScheduleConfig(name=LatentBatchAlignmentSchedule.name),
        ScheduleConfig(name=ReconstructionBatchAlignmentSchedule.name),
        ScheduleConfig(name=TranslationSchedule.name),
        ScheduleConfig(name=ClassificationSchedule.name),
    ]


class UnsupervisedGroupIdentificationTask(BaseTask):
    name = "unsupervised_group_identification"
    train_schedule_configs = [
        ScheduleConfig(name=TranslationSchedule.name),
        ScheduleConfig(name=ClusteringSchedule.name),
    ]

    finetune_schedule_configs = [
        ScheduleConfig(name=TranslationFinetuneSchedule.name),
        ScheduleConfig(name=ClusteringFinetuneSchedule.name),
    ]

    transfer_schedule_configs = [
        ScheduleConfig(name=LatentBatchAlignmentSchedule.name),
        ScheduleConfig(name=ReconstructionBatchAlignmentSchedule.name),
        ScheduleConfig(name=TranslationSchedule.name),
        ScheduleConfig(name=ClusteringSchedule.name),
    ]


class TaskManager(ObjectManager):
    """\
    Task

    Each task manages its corresponding schedules. The base task handles only evaluation and inference.
    """

    name = "tasks"
    constructors = [
        CrossModelPredictionTask,
        SupervisedGroupIdentificationTask,
        UnsupervisedGroupIdentificationTask,
    ]

    @staticmethod
    def get_task_by_name_or_config(task):
        if isinstance(task, str):
            return TaskManager.get_constructor_by_name(task)()
        else:
            return CustomizedTask()
