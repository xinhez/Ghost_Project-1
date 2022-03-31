from copyreg import constructor
from utils import AlternativelyNamedObject, ObjectManager


class BaseTask(AlternativelyNamedObject):
    pass


class CrossModelPredictionTask(BaseTask):
    name = 'cross_model_prediction'


class SupervisedGroupIdentificationTask(BaseTask):
    name = 'supervised_group_identification'


class UnsupervisedGroupIdentificationTask(BaseTask):
    name = 'unsupervised_group_identification'


class TaskManager(ObjectManager):
    constructors = [CrossModelPredictionTask, SupervisedGroupIdentificationTask, UnsupervisedGroupIdentificationTask]