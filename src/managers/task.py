from src.managers.base import AlternativelyNamedObject, ObjectManager
from src.managers.schedule import ClassificationSchedule, ClusteringSchedule, TranslationSchedule
from src.utils import average_dictionary_values_by_count, combine_tensor_lists, combine_value_dictionaries


class BaseTask(AlternativelyNamedObject):
    name = 'inference' 


    def evaluate_outputs(self, modalities, labels, outputs):
        raise Exception("Not Implemented!")

    
    def run_by_batch(self, model, dataloader, Schedule=None, evaluate_model=False):
        all_outputs = []
        all_losses  = {}

        for modalities, *batches_and_maybe_labels in dataloader:
            if len(batches_and_maybe_labels) == 1:
                batches, labels = batches_and_maybe_labels, None
            elif len(batches_and_maybe_labels) == 2:
                batches, labels = batches_and_maybe_labels

            outputs = model(modalities, batches, labels)
            all_outputs = combine_tensor_lists(all_outputs, outputs)
            
            if Schedule is not None:
                losses = Schedule(model).step(model)
                all_losses[Schedule.name] = combine_value_dictionaries(all_losses, losses)

        if evaluate_model:
            self.evaluate_outputs(dataloader.dataset.modalities, dataloader.dataset.labels, all_outputs)

        all_losses = average_dictionary_values_by_count(all_losses, len(dataloader.dataset))
        return all_outputs


    def evaluate(self, model, data_eval): 
        dataloader_eval = data_eval.create_dataloader(model)
        model.eval() 
        self.run_by_batch(model, dataloader_eval, evaluate_model=True)


    def infer(self, model, data_infer): 
        dataloader_infer = data_infer.create_dataloader(model)
        model.eval()
        latents, fused_latents, translations, cluster_outputs = self.run_by_batch(model, dataloader_infer)
        raise Exception("Re-infer not implemented!")

    
    def train(self, model, data): 
        dataloader = data.create_dataloader(model)

        model.train()
        for Schedule in self.Schedules:
            self.run_by_batch(model, dataloader, Schedule)


    def transfer(self, model, data, data_transfer): 
        raise Exception("Not Implemented!")


class CrossModelPredictionTask(BaseTask):
    name = 'cross_model_prediction'    
    def __init__(self): 
        self.Schedules = [ClassificationSchedule, TranslationSchedule]


class SupervisedGroupIdentificationTask(BaseTask):
    name = 'supervised_group_identification'
    def __init__(self): 
        self.Schedules = [TranslationSchedule, ClassificationSchedule]


class UnsupervisedGroupIdentificationTask(BaseTask):
    name = 'unsupervised_group_identification'
    def __init__(self): 
        self.Schedules = [TranslationSchedule, ClusteringSchedule]


class TaskManager(ObjectManager):
    """\
    Task

    Each task manages its corresponding schedules. The base task handles only evaluation and inference.
    """
    name = 'tasks'
    constructors = [BaseTask, CrossModelPredictionTask, SupervisedGroupIdentificationTask, UnsupervisedGroupIdentificationTask]