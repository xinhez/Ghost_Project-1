import numpy as np

from managers.task import TaskManager


def infer(model, data_infer):
    """\
    infer
    """
    model.data_batch_encoder.fit(data_infer.batches)
    
    dataset_infer = data_infer.create_dataset(model)

    model.eval()
    raise Exception('Not implemented!')


def evaluate(model, data_eval):
    """\
    evaluate
    """
    model.data_label_encoder.fit(data_eval.labels)
    model.data_batch_encoder.fit(data_eval.batches)

    dataset_eval = data_eval.create_dataset(model)

    model.eval()
    raise Exception('Not implemented!')


def train(task, model, data, data_eval): 
    """\
    train
    """
    model.data_label_encoder.fit(data.labels  + data_eval.labels)
    model.data_batch_encoder.fit(data.batches + data_eval.batches)

    dataset      = data.create_dataset(model)
    dataset_eval = data_eval.create_dataset(model)

    task_manager = TaskManager.get_constructor_by_name(task)(train.__name__)

    model.train()
    raise Exception("Not Implemented!")


def transfer(task, model, data, data_transfer, data_eval):
    """\
    transfer
    """
    model.data_label_encoder.fit(data.labels  + data_eval.labels)
    model.data_batch_encoder.fit(data.batches + data_transfer.batches + data_eval.batches)

    dataset          = data.create_dataset(model)
    dataset_transfer = data_transfer.create_dataset(model)
    dataset_eval     = data_eval.create_dataset(model)

    task_manager = TaskManager.get_constructor_by_name(task)(transfer.__name__)

    model.train()
    raise Exception("Not Implemented!")