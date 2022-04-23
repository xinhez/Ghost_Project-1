import torch
import torch.utils.data as D

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, r2_score
from src.managers.data import DataManager

import src.utils as utils

from src.managers.schedule import ClassificationSchedule, ScheduleManager


class Task:
    name = "Customized"

    def update_schedules(
        self, logger, model, learning_rate, schedule_configs, model_path, method
    ):
        if schedule_configs is None:
            raise Exception(f"Please provide {method}_schedules.")
        self.schedules = [
            ScheduleManager.get_constructor_by_name(config.name)(
                logger,
                model,
                learning_rate,
                config,
                model_path,
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

        for modalities, batches, labels in dataloader:
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
                utils.inplace_combine_tensor_lists(all_outputs, outputs)

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

        return utils.concat_tensor_lists(all_outputs)

    def evaluate_outputs(self, logger, dataloader, outputs):
        if not isinstance(dataloader.sampler, D.SequentialSampler):
            raise Exception(
                "Please only evaluate outputs with non-shuffling dataloader."
            )
        dataset = dataloader.dataset
        labels = dataset.labels
        translations_outputs, predictions, *_ = outputs

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
        metrics = self.evaluate_outputs(logger, dataloader_eval, outputs)

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
        dataloader = data.create_dataloader(
            model, shuffle=True, batch_size=batch_size, random_seed=random_seed
        )
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
        schedules,
        model,
        data,
        data_validate,
        batch_size,
        n_epoch,
        learning_rate,
        logger,
        model_path,
        save_best_model,
        checkpoint,
        writer,
        random_seed,
    ):
        logger.log_method_start(self.train.__name__)

        self.update_schedules(
            logger, model, learning_rate, schedules, model_path, self.train.__name__,
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
        schedules,
        model,
        data,
        data_validate,
        batch_size,
        n_epoch,
        learning_rate,
        logger,
        model_path,
        save_best_model,
        checkpoint,
        writer,
        random_seed,
    ):
        logger.log_method_start(self.finetune.__name__)

        self.update_schedules(
            logger, model, learning_rate, schedules, model_path, self.finetune.__name__,
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
        schedules,
        model,
        data,
        data_transfer,
        data_validate,
        batch_size,
        n_epoch,
        learning_rate,
        logger,
        model_path,
        save_best_model,
        checkpoint,
        writer,
        random_seed,
    ):
        logger.log_method_start(self.transfer.__name__)

        self.update_schedules(
            logger, model, learning_rate, schedules, model_path, self.transfer.__name__,
        )

        dataloader = data.create_dataloader(
            model, shuffle=True, batch_size=batch_size, random_seed=random_seed
        )
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
