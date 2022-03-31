from managers.schedule import TaskManager

def infer(mode, data_infer):
    """\
    infer
    """
    raise Exception('Not implemented!')


def evaluate(mode, data_eval):
    """\
    evaluate
    """
    raise Exception('Not implemented!')


def train(task, model, data, data_eval): 
    """\
    train
    """
    task_manager = TaskManager(task)
    task_manager.train(model, data, data_eval)


def transfer(self, model, data, data_transfer, data_eval):
    """\
    transfer
    """
    raise Exception("Not Implemented!")