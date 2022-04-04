from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score

from src.config import ScheduleConfig
from src.logger import Logger
from src.managers.base import AlternativelyNamedObject, ObjectManager
from src.managers.schedule import ClassificationSchedule, ClusteringSchedule, ScheduleManager, TranslationSchedule
from src.utils import average_dictionary_values_by_count, combine_tensor_lists, sum_value_dictionaries


class CustomizedTask(AlternativelyNamedObject):
    name = 'customized' 


    def update_schedules(self, model, schedule_configs):
        if schedule_configs is not None:
            self.schedules = [
                ScheduleManager.get_constructor_by_name(config.name)(model, config) 
                for config in schedule_configs
            ]


    def evaluate_outputs(self, logger, dataset, outputs):
        labels = dataset.labels        
        _, _, translations, predictions = outputs

        labels      = labels.detach().numpy()
        predictions = predictions.detach().numpy()
        metrics = {
            'ari': adjusted_rand_score(labels, predictions),
            'nmi': normalized_mutual_info_score(labels, predictions),
        }

        logger.log_evaluation_metrics(metrics)
        return metrics

    
    def run_through_data(self, logger, model, dataloader, schedule=None, evaluate_outputs=False):
        all_outputs = []
        all_losses  = {}

        for modalities, *batches_and_maybe_labels in dataloader:
            if len(batches_and_maybe_labels) == 1:
                batches, labels = *batches_and_maybe_labels, None
            elif len(batches_and_maybe_labels) == 2:
                batches, labels = batches_and_maybe_labels

            outputs = model(modalities, batches, labels)
            all_outputs = combine_tensor_lists(all_outputs, outputs)
            
            if schedule is not None:
                losses = schedule.step(model)
                all_losses[schedule.name] = sum_value_dictionaries(all_losses, losses)

        if all_losses: 
            all_losses = average_dictionary_values_by_count(all_losses, len(dataloader.dataset))
            logger.log_losses(all_losses)

        raise Exception("Logic for best head not implemented.")
        metrics = self.evaluate_outputs(logger, dataloader.dataset, all_outputs) if evaluate_outputs else {}

        return all_outputs, metrics


    def evaluate(self, model, data_eval, batch_size, save_log_path): 
        logger = Logger(save_log_path)
        
        dataloader_eval = data_eval.create_dataloader(model, batch_size)
        
        model.eval() 

        _, metrics = self.run_through_data(logger, model, dataloader_eval, evaluate_outputs=True)

        return metrics


    def infer(self, model, data_infer, batch_size, save_log_path): 
        logger = Logger(save_log_path)

        dataloader_infer = data_infer.create_dataloader(model, batch_size)

        model.eval()

        outputs, _ = self.run_through_data(logger, model, dataloader_infer)

        raise Exception("Re-infer not implemented!")
        return outputs

    
    def train(self, model, data, batch_size, n_epoch, schedule_configs, save_log_path): 
        logger = Logger(save_log_path)

        self.update_schedules(model, schedule_configs)

        dataloader = data.create_dataloader(model, batch_size)

        model.train()
        for epoch in range(n_epoch):
            logger.log_epoch_start(epoch, self.train.__name__)

            for schedule in self.schedules:
                logger.log_schedule_start(schedule)

                self.run_through_data(logger, model, dataloader, schedule)


    def transfer(self, model, data, data_transfer, n_epoch, schedule_configs, save_log_path): 
        logger = Logger(save_log_path)

        self.update_schedules(model, schedule_configs)

        for epoch in range(n_epoch):
            logger.log_epoch_start(epoch, self.transfer.__name__)
            for schedule in self.schedules:
                logger.log_schedule_start(schedule)
                raise Exception("Not Implemented!")
                self.run_through_data(logger, model, dataloader, schedule)


class CrossModelPredictionTask(CustomizedTask):
    name = 'cross_model_prediction'    
    def train(self, model, data, batch_size, n_epoch, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=ClassificationSchedule.name),
            ScheduleConfig(name=TranslationSchedule.name),
        ]
        super().train(model, data, batch_size, n_epoch, schedule_configs, *args)


class SupervisedGroupIdentificationTask(CustomizedTask):
    name = 'supervised_group_identification'
    def train(self, model, data, batch_size, n_epoch, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=TranslationSchedule.name),
            ScheduleConfig(name=ClassificationSchedule.name),
        ]
        super().train(model, data, batch_size, n_epoch, schedule_configs, *args)


class UnsupervisedGroupIdentificationTask(CustomizedTask):
    name = 'unsupervised_group_identification'
    def train(self, model, data, batch_size, n_epoch, _, *args): 
        schedule_configs = [
            ScheduleConfig(name=TranslationSchedule.name),
            ScheduleConfig(name=ClusteringSchedule.name),
        ]
        super().train(model, data, batch_size, n_epoch, schedule_configs, *args)


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