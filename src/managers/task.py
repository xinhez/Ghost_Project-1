from managers.base import AlternativelyNamedObject, ObjectManager


class BaseTask(AlternativelyNamedObject):
    def __init__(self, _): pass


class CrossModelPredictionTask(BaseTask):
    name = 'cross_model_prediction'


class SupervisedGroupIdentificationTask(BaseTask):
    name = 'supervised_group_identification'


class UnsupervisedGroupIdentificationTask(BaseTask):
    name = 'unsupervised_group_identification'


class TaskManager(ObjectManager):
    name = 'tasks'
    constructors = [CrossModelPredictionTask, SupervisedGroupIdentificationTask, UnsupervisedGroupIdentificationTask]