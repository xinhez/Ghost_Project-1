import torch

from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from src.managers.data import DataManager

import src.utils as utils

from src.config import ScheduleConfig
from src.managers.base import AlternativelyNamedObject, ObjectManager
from src.managers.schedule import ScheduleManager
from src.managers.schedule import ClassificationSchedule, ClusteringSchedule, TranslationSchedule
from src.managers.schedule import LatentBatchAlignmentSchedule, ReconstructionBatchAlignmentSchedule


class CustomizedTask(AlternativelyNamedObject):
    name = 'customized' 


    def update_schedules(self, logger, model, schedule_configs, save_model_path, method):
        if schedule_configs is not None:
            self.schedules = [
                ScheduleManager.get_constructor_by_name(config.name)(
                    logger, model, config, save_model_path, self.get_short_name(), method, schedule_order
                ) 
                for (schedule_order, config) in enumerate(schedule_configs)
            ]

    
    def run_through_data(
            self, logger, model, dataloader, schedule=None, 
            train_model=False, infer_model=False,
            save_best_model=False, checkpoint_model_name=None,
        ):
        """\
        Losses can only computed by the schedule.
        Inference involves moving tensors to CPU and can be slow.
        """
        if train_model and schedule is None:
            raise Exception("Please provide training schedules.")
        
        all_outputs = []
        all_losses  = {}

        for modalities, *batches_and_maybe_labels in dataloader:
            if len(batches_and_maybe_labels) == 1:
                batches, labels = *batches_and_maybe_labels, None
            elif len(batches_and_maybe_labels) == 2:
                batches, labels = batches_and_maybe_labels

            outputs = model(modalities, batches, labels)
            
            if schedule is not None:
                losses = schedule.step(model, train_model)
                losses = utils.amplify_value_dictionary_by_sample_size(losses, len(batches))
                all_losses = utils.sum_value_dictionaries(all_losses, losses)
                
            if infer_model:
                outputs = utils.move_tensor_list_to_cpu(outputs)
                all_outputs = utils.combine_tensor_lists(all_outputs, outputs)

        if all_losses: 
            all_losses = utils.average_dictionary_values_by_sample_size(all_losses, len(dataloader.dataset))
            logger.log_losses(all_losses)

        if save_best_model and schedule.check_and_update_best_loss(all_losses):
            schedule.save_model(model)

        if checkpoint_model_name is not None:
            schedule.save_model(model, checkpoint_model_name)

        return all_outputs


    def evaluate_outputs(self, logger, dataset, outputs):
        labels = dataset.labels        
        translations, cluster_outputs, *_ = outputs
        predictions = cluster_outputs.argmax(axis=1)

        # raise Exception("translations Not Implemented!")
        accuracy = torch.sum(labels == predictions).data / predictions.shape[0]

        labels      = labels.numpy()
        predictions = predictions.numpy()
        metrics = {
            'acc': accuracy,
            'ari': adjusted_rand_score(labels, predictions),
            'nmi': normalized_mutual_info_score(labels, predictions, average_method="geometric"),
        }

        logger.log_metrics(metrics)
        return metrics


    def evaluate(self, model, data_eval, logger): 
        logger.log_method_start(self.evaluate.__name__)
        
        dataloader_eval = data_eval.create_dataloader(model, shuffle=False)
        
        model.eval() 
        with torch.no_grad():
            outputs = self.run_through_data(logger, model, dataloader_eval, infer_model=True)
        metrics = self.evaluate_outputs(logger, dataloader_eval.dataset, outputs)

        return metrics


    def infer(self, model, data_infer, logger, modalities_provided): 
        logger.log_method_start(self.infer.__name__)

        dataloader_infer = data_infer.create_dataloader(model, shuffle=False)

        model.eval()

        with torch.no_grad():
            outputs = self.run_through_data(logger, model, dataloader_infer, infer_model=True)

        if len(modalities_provided) == 0 or data_infer.n_modality == len(modalities_provided):
            return DataManager.anndata_from_outputs(model, dataloader_infer.dataset, outputs)

        raise Exception("Re-infer not implemented!")


    def train(
        self, schedule_configs, model, data, data_validation, batch_size, n_epoch, 
        logger, save_model_path, save_best_model, checkpoint,
        ): 
        logger.log_method_start(self.train.__name__, self.name)

        self.update_schedules(logger, model, schedule_configs, save_model_path, self.train.__name__)

        dataloader = data.create_dataloader(model, shuffle=True, batch_size=batch_size)
        datalodaer_validation = data_validation.create_dataloader(model, shuffle=False, batch_size=batch_size)

        model.train()
        for epoch in range(n_epoch):
            epoch += 1
            logger.log_epoch_start(epoch, n_epoch)

            for schedule in self.schedules:
                logger.log_schedule_start(schedule)

                self.run_through_data(logger, model, dataloader, schedule, train_model=True)

                if checkpoint > 0:
                    if epoch % checkpoint == 0:
                        checkpoint_model_name = f'epoch_{epoch}.pt'
                else:
                    checkpoint_model_name = None

                with torch.no_grad():
                    self.run_through_data(
                        logger, model, datalodaer_validation, schedule,
                        save_best_model=save_best_model, checkpoint_model_name=checkpoint_model_name,
                    )


    def transfer(
        self, schedule_configs, model, data, data_transfer, data_validation, batch_size, n_epoch, 
        logger, save_model_path, save_best_model, checkpoint,
    ): 
        logger.log_method_start(self.transfer.__name__, self.name)

        self.update_schedules(logger, model, schedule_configs, save_model_path, self.transfer.__name__)

        dataloader = data.create_dataloader(model, shuffle=True, batch_size=batch_size)
        dataloader_train_and_transfer = data.create_joint_dataloader(
            data_transfer, model, shuffle=True, batch_size=batch_size
        )
        datalodaer_validation = data_validation.create_dataloader(model, shuffle=False, batch_size=batch_size)

        for epoch in range(n_epoch):
            epoch += 1
            logger.log_epoch_start(epoch, n_epoch)
            for schedule in self.schedules:
                logger.log_schedule_start(schedule)
                if isinstance(schedule, ClassificationSchedule):
                    self.run_through_data(logger, model, dataloader, schedule, train_model=True)
                else:
                    self.run_through_data(logger, model, dataloader_train_and_transfer, schedule, train_model=True)

                if checkpoint > 0:
                    if epoch % checkpoint == 0:
                        checkpoint_model_name = f'epoch_{epoch}.pt'
                else:
                    checkpoint_model_name = None

                with torch.no_grad():
                    self.run_through_data(
                        logger, model, datalodaer_validation, schedule,
                        save_best_model=save_best_model, checkpoint_model_name=checkpoint_model_name,
                    )


class CrossModelPredictionTask(CustomizedTask):
    name = 'cross_model_prediction'    
    def train(self, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=ClassificationSchedule.name),
            ScheduleConfig(name=TranslationSchedule.name),
        ]
        super().train(schedule_configs, *args)


    def transfer(self, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=LatentBatchAlignmentSchedule.name), 
            ScheduleConfig(name=ReconstructionBatchAlignmentSchedule.name),
            ScheduleConfig(name=ClassificationSchedule.name),
            ScheduleConfig(name=TranslationSchedule.name),
        ]
        super().transfer(schedule_configs, *args)


class SupervisedGroupIdentificationTask(CustomizedTask):
    name = 'supervised_group_identification'
    def train(self, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=TranslationSchedule.name),
            ScheduleConfig(name=ClassificationSchedule.name),
        ]
        super().train(schedule_configs, *args)


    def transfer(self, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=LatentBatchAlignmentSchedule.name), 
            ScheduleConfig(name=ReconstructionBatchAlignmentSchedule.name),
            ScheduleConfig(name=TranslationSchedule.name),
            ScheduleConfig(name=ClassificationSchedule.name),
        ]
        super().transfer(schedule_configs, *args)


class UnsupervisedGroupIdentificationTask(CustomizedTask):
    name = 'unsupervised_group_identification'
    def train(self, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=TranslationSchedule.name),
            ScheduleConfig(name=ClusteringSchedule.name),
        ]
        super().train(schedule_configs, *args)


    def transfer(self, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=LatentBatchAlignmentSchedule.name), 
            ScheduleConfig(name=ReconstructionBatchAlignmentSchedule.name),
            ScheduleConfig(name=TranslationSchedule.name),
            ScheduleConfig(name=ClusteringSchedule.name),
        ]
        super().transfer(schedule_configs, *args)


class TaskManager(ObjectManager):
    """\
    Task

    Each task manages its corresponding schedules. The base task handles only evaluation and inference.
    """
    name = 'tasks'
    constructors = [
        CustomizedTask, 
        CrossModelPredictionTask, SupervisedGroupIdentificationTask, UnsupervisedGroupIdentificationTask,
    ]