import numpy as np

"""
validate_anndatas_counts
"""
def validate_anndatas_counts(adatas):
    n_counts = [len(adata) for adata in adatas]
    n_uniques = np.unique(n_counts)
    if len(n_uniques) > 1:
        raise Exception("Please provide equal numbers of samples for all modalities.")

"""
validate_anndatas_labels
"""
def validate_anndatas_labels(reference, obs_key_label):
    if obs_key_label not in reference.obs_keys():
        raise Exception(f"Please provide labels at obs['{obs_key_label}'] for the first modality.")

"""
validate_anndatas
"""
def validate_anndatas(adatas, reference_index_label, obs_key_label):
    if len(adatas) < 2: 
        raise Exception("Please provide at least 2 modalities.")

    validate_anndatas_counts(adatas)
    validate_anndatas_labels(adatas[reference_index_label], obs_key_label)