import numpy as np

from utils import count_unique


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
    reference_index_label, reference_index_batch, 
    obs_key_label, obs_key_batch
):
    """\
    Format the AnnDatas into modalities, labels and batches.
    """
    modalities = [adata.X for adata in adatas]
    labels = adatas[reference_index_label].obs[obs_key_label]
    batches = get_batch(adatas[reference_index_batch], obs_key_batch, 0)

    return modalities, labels, batches


# ==================== AnnData Validation ====================
def validate_anndatas_counts(adatas):
    """\
    validate_anndatas_counts
    """
    n_counts = [len(adata) for adata in adatas]
    if count_unique(n_counts) > 1:
        raise Exception("Please provide equal numbers of samples for all modalities.")


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
    validate_anndatas_labels(adatas[reference_index_label], obs_key_label)


# ==================== Size Extraction ====================
def get_input_sizes(modalities):
    """\
    Compute the input sizes for model configurations based on the given modalities.
    """
    return [modality.shape[1] for modality in modalities]


def get_output_size(labels):
    """\
    Compute the output size for model configuration based on the given labels.
    """
    return count_unique(labels)


def get_batch_count(batches):
    """\
    Count the number of unique batches for the given batches.
    """
    return count_unique(batches)


def get_model_config_sizes(
    adatas, 
    reference_index_label, reference_index_batch, 
    obs_key_label, obs_key_batch
):
    """
    Compute the sizes for model configuration based on the given dataset.
    """
    modalities, labels, batches = format_data(
        adatas, 
        reference_index_label, reference_index_batch, 
        obs_key_label, obs_key_batch
    )

    input_sizes = get_input_sizes(modalities)
    output_size = get_output_size(labels)
    n_batch     = get_batch_count(batches)
    
    return input_sizes, output_size, n_batch