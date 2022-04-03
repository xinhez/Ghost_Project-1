from managers.base import AlternativelyNamedObject, ObjectManager


class BaseTask(AlternativelyNamedObject):
    def evaluate(self, model, data_eval): pass 


    def infer(self, model, data_infer): pass

    
    def train(self, model, data): pass 


    def transfer(self, model, data, data_transfer): pass 


class CrossModelPredictionTask(BaseTask):
    name = 'cross_model_prediction'
    def __init__(self, function): pass


class SupervisedGroupIdentificationTask(BaseTask):
    name = 'supervised_group_identification'
    def __init__(self, function): pass


class UnsupervisedGroupIdentificationTask(BaseTask):
    name = 'unsupervised_group_identification'
    def __init__(self, function): pass


class TaskManager(ObjectManager):
    name = 'tasks'

    function_train = 'train'
    function_transfer = 'transfer'

    constructors = [CrossModelPredictionTask, SupervisedGroupIdentificationTask, UnsupervisedGroupIdentificationTask]


    def get_constructor_by_name(name=None):
        if name is None:
            return BaseTask
        else:
            return super().get_constructor_by_name(name)