import numpy as np

from managers.task import TaskManager

def run_by_batch(model, dataloader):
    for modalities, *batches_and_maybe_labels in dataloader: 
        if len(batches_and_maybe_labels) == 2:
            batches, labels = batches_and_maybe_labels
        else:
            batches = batches_and_maybe_labels
    print(batches)
    print(labels)
    raise Exception("Not Implemented!")


def infer(model, data_infer):
    """\
    infer
    """
    model.data_batch_encoder.fit(data_infer.batches)
    
    dataloader_infer = data_infer.create_dataloader(model)

    model.eval()
    raise Exception('Not implemented!')


def evaluate(model, data_eval):
    """\
    evaluate
    """
    model.data_label_encoder.fit(data_eval.labels)
    model.data_batch_encoder.fit(data_eval.batches)

    dataloader_eval = data_eval.create_dataloader(model)

    model.eval()
    raise Exception('Not implemented!')


def train(task, model, data, data_eval): 
    """\
    train
    """
    model.data_label_encoder.fit(data.labels  + data_eval.labels)
    model.data_batch_encoder.fit(data.batches + data_eval.batches)

    dataloader      = data.create_dataloader(model)
    dataloader_eval = data_eval.create_dataloader(model)

    task_manager = TaskManager.get_constructor_by_name(task)(train.__name__)

    model.train()
    raise Exception("Not Implemented!")


def transfer(task, model, data, data_transfer, data_eval):
    """\
    transfer
    """
    model.data_label_encoder.fit(data.labels  + data_eval.labels)
    model.data_batch_encoder.fit(data.batches + data_transfer.batches + data_eval.batches)

    dataloader          = data.create_dataloader(model)
    dataloader_transfer = data_transfer.create_dataloader(model)
    dataloader_eval     = data_eval.create_dataloader(model)

    task_manager = TaskManager.get_constructor_by_name(task)(transfer.__name__)

    model.train()
    raise Exception("Not Implemented!")