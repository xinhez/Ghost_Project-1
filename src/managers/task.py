from managers.base import AlternativelyNamedObject, ObjectManager
from managers.schedule import ClassificationSchedule, ClusteringSchedule, TranslationSchedule
from utils import combine_tensor_lists


class BaseTask(AlternativelyNamedObject):
    name = 'inference' 

    
    def run_by_batch(self, model, dataloader, Schedule=None):
        all_latents, all_fused_latents, all_translations, all_cluster_outputs = [], [], [], []
        for modalities, *batches_and_maybe_labels in dataloader:
            if len(batches_and_maybe_labels) == 1:
                batches = batches_and_maybe_labels
            elif len(batches_and_maybe_labels) == 2:
                batches, labels = batches_and_maybe_labels

            latents, fused_latents, translations, cluster_outputs = model(modalities)
            all_latents, all_fused_latents, all_translations, all_cluster_outputs = combine_tensor_lists(
                (all_latents, all_fused_latents, all_translations, all_cluster_outputs),
                (latents,     fused_latents,     translations,     cluster_outputs),
            )
            
            if Schedule is not None:
                Schedule(model).step(modalities, labels, translations, cluster_outputs)

        return latents, fused_latents, translations, cluster_outputs


    def evaluate(self, model, data_eval): 
        dataloader_eval = data_eval.create_dataloader(model)
        model.eval() 
        latents, fused_latents, translations, cluster_outputs = self.run_by_batch(model, dataloader_eval)


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