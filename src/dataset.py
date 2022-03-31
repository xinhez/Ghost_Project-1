import numpy as np

from constants import TRAIN
from utils import count_unique


class RawData():
    def __init__(self, modalities, labels, batches):
        self.modalities = modalities
        self.labels = labels,
        self.batches = batches


    @property
    def input_sizes(self):
        return [modality.shape[1] for modality in self.modalities]

    
    @property
    def output_size(self):
        return count_unique(self.labels)

    
    @property
    def n_batch(self):
        return count_unique(self.batches)


# ==================== AnnData Format ====================
def get_batch(adata, obs_key_batch, default_batch):
    """\
    Get batch information for the given AnnData or return the given default batch if not already exists.
    """
    if obs_key_batch not in adata.obs_keys():
        return np.repeat(default_batch, len(adata))
    else:
        return adata.obs[obs_key_batch]


def format_data(
    adatas, 
    reference_index_label, obs_key_label, 
    reference_index_batch, obs_key_batch,
    default_batch=TRAIN
):
    """\
    Format the AnnDatas into RawData object containing modalities, labels and batches.
    """
    validate_anndatas(adatas, obs_key_label, reference_index_label)

    modalities = [adata.X for adata in adatas]
    if reference_index_label is None:
        labels = []
    else:
        labels = adatas[reference_index_label].obs[obs_key_label]
    batches = get_batch(adatas[reference_index_batch], obs_key_batch, default_batch)

    return RawData(modalities, labels, batches)


# ==================== AnnData Validation ====================
def validate_anndatas_counts(adatas):
    """\
    validate_anndatas_counts
    """
    n_counts = [len(adata) for adata in adatas]
    if count_unique(n_counts) > 1:
        raise Exception("Please provide equal numbers of samples for all modalities.")


def validate_anndatas_dimensions(adatas):
    """\
    Validate the given AnnDatas are all 2D (number of samples X number of features).
    """
    for i, adata in enumerate(adatas):
        if len(adata.shape) != 2:
            raise Exception(f"The {i+1}-th modality in the given dataset is not 2D.")


def validate_anndatas_labels(reference, obs_key_label):
    """\
    validate_anndatas_labels
    """
    if obs_key_label not in reference.obs_keys():
        raise Exception(f"Please provide labels at obs['{obs_key_label}'] for the first modality.")


def validate_anndatas(adatas, obs_key_label, reference_index_label):
    """\
    validate_anndatas
    """
    if len(adatas) < 2: 
        raise Exception("Please provide at least 2 modalities.")

    validate_anndatas_counts(adatas)
    validate_anndatas_dimensions(adatas)
    validate_anndatas_labels(adatas[reference_index_label], obs_key_label)


# ==================== Size Extraction ====================
def validate_data_sizes_match(data, data_eval):
    """\
    Validate that the training dataset and the evaluation dataset have the same input sizes.
    """
    if data.input_sizes != data_eval.input_sizes:
        raise Exception("All feature dimensions in the evaluation dataset must match with the registered data.")