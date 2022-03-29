import numpy as np

from config import create_default_config

"""
autofill_batch
"""
def autofill_batch(reference, obs_key_batch, autofill_batch):
    if obs_key_batch not in reference.obs_keys():
        return np.repeat(autofill_batch, len(reference))
    else:
        return referense.obs[obs_key_batch]

"""
autofill_config
"""
def autofill_config(modalities, labels, batches):
    input_sizes = [modality.shape[1] for modality in modalities]
    output_size = len(np.unique(labels))
    n_batch = len(np.unique(batches))

    return create_default_config(input_sizes, output_size, n_batch)