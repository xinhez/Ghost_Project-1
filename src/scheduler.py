from constants import BATCH_ALIGNMENT, CLASSIFICATION, CLUSTERING, TRANSLATION
from constants import TASKS, CROSS_MODEL_PREDICTION, SUPERVISED_GROUP_IDENTIFICATION, UNSUPERVISED_GROUP_IDENTIFICATION


class Schedule():
    @staticmethod
    def schedule_to_losses(schedule):
        return {
            BATCH_ALIGNMENT: [],
            CLASSIFICATION:  [],
            CLUSTERING:      [],
            TRANSLATION:     [],
        }[schedule]


class Scheduler():
    def __init__(self, task):
        schedules_by_task = {
            CROSS_MODEL_PREDICTION: [CLASSIFICATION, TRANSLATION],
            SUPERVISED_GROUP_IDENTIFICATION: [TRANSLATION, CLASSIFICATION],
            UNSUPERVISED_GROUP_IDENTIFICATION: [TRANSLATION, CLUSTERING],
        }
        self.schedules = schedules_by_task[task]


def infer(mode, data_eval):
    """\
    infer
    """
    raise Exception('Not implemented!')


def evaluate(mode, data_eval):
    """\
    evaluate
    """
    raise Exception('Not implemented!')


def train(model, task, data, data_eval): 
    """\
    train
    """
    if task not in TASKS:
        raise Exception(f"Only {TASKS} tasks are supported.")
    scheduler = Scheduler(task)

    raise Exception('Not implemented!')