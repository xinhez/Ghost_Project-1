from sklearn import cluster
from managers.base import AlternativelyNamedObject, ObjectManager
from managers.schedule import ClassificationSchedule, ClusteringSchedule, TranslationSchedule


class BaseTask(AlternativelyNamedObject):
    name = 'inference'
    def __init__(self):
        self.schedules = []

    
    def run_by_batch(self, model, dataloader, schedule=None):
        all_latents = []
        all_fused_latents = []
        all_translations = []
        all_cluster_outputs=  []
        for modalities, *batches_and_maybe_labels in dataloader:
            if len(batches_and_maybe_labels) == 1:
                batches = batches_and_maybe_labels
            elif len(batches_and_maybe_labels) == 2:
                batches, labels = batches_and_maybe_labels
            latents, fused_latents, translations, cluster_outputs = model(modalities)
            if schedule is not None:
                schedule.step(modalities, labels, translations, cluster_outputs)


    def evaluate(self, model, data_eval): 
        model.eval() 


    def infer(self, model, data_infer): 
        model.eval()

    
    def train(self, model, data): 
        dataloader = data.create_dataloader(model)
        model.train()
        for schedule in self.schedules:
            self.run_by_batch(model, dataloader, schedule)


    def transfer(self, model, data, data_transfer): 
        model.train()
        for schedule in self.schedules:
            raise Exception("Combine data not implemented!")
            self.run_by_batch(model, data, schedule)


class CrossModelPredictionTask(BaseTask):
    name = 'cross_model_prediction'    
    def __init__(self): 
        self.schedules = [ClassificationSchedule.name, TranslationSchedule.name]


class SupervisedGroupIdentificationTask(BaseTask):
    name = 'supervised_group_identification'
    def __init__(self): 
        self.schedules = [TranslationSchedule.name, ClassificationSchedule.name]


class UnsupervisedGroupIdentificationTask(BaseTask):
    name = 'unsupervised_group_identification'
    def __init__(self): 
        self.schedules = [TranslationSchedule.name, ClusteringSchedule.name]


class TaskManager(ObjectManager):
    """\
    Task

    Each task manages its corresponding schedules. The base task handles only evaluation and inference.
    """
    name = 'tasks'
    constructors = [BaseTask, CrossModelPredictionTask, SupervisedGroupIdentificationTask, UnsupervisedGroupIdentificationTask]