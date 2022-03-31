from cProfile import label
from utils import count_unique

class Data():
    name = 'Data'

    def __init__(self, modalities, batches_or_batch, labels_or_None):
        self.modalities = modalities
        self.save_batches(batches_or_batch)
        self.labels     = labels_or_None


    @property
    def input_sizes(self):
        return [modality.shape[1] for modality in self.modalities]

    
    @property
    def output_size(self):
        return count_unique(self.labels)

    
    @property
    def n_batches(self):
        return count_unique(self.batches)


    @property
    def n_samples(self):
        return len(self.modalities[0])


    def save_batches(self, batches_or_batch):
        if isinstance(batches_or_batch, list):
            self.batches = batches_or_batch
        else:
            self.batches = [batches_or_batch for _ in range(self.n_samples)]


class DataManager():
    evaluate = 'evaluate'
    infer = 'infer'
    train = 'train'
    transfer = 'transfer'
    groups = [evaluate, infer, train, transfer]

    label = 'label'
    batch = 'batch'
    keys = [label, batch]

    @staticmethod
    def format_anndatas(group, adatas, batch_index, batch_key, label_index=None, label_key=None):
        DataManager.validate_anndatas(group, adatas, batch_index, batch_key, label_index, label_key)

        modalities       = DataManager.get_Xs_from_anndatas(adatas)
        batches_or_batch = DataManager.get_obs_from_anndatas(adatas, batch_index, batch_key, group)
        labels_or_None   = DataManager.get_obs_from_anndatas(adatas, label_index, label_key)
        
        return Data(modalities, batches_or_batch, labels_or_None)


    @staticmethod
    def validate_anndatas(group, adatas, batch_index, batch_key, label_index, label_key):
        DataManager.validate_anndatas_exist(group, adatas)
        DataManager.validate_anndatas_sample_sizes(adatas)
        DataManager.validate_anndatas_dimensions(adatas)

        DataManager.validate_index_and_key(adatas, batch_index, batch_key)
        DataManager.validate_index_and_key(adatas, label_index, label_key)


    @staticmethod
    def validate_anndatas_exist(group, adatas):
        if adatas is None:
            raise Exception(f"Please provide AnnData for {group}.")


    @staticmethod
    def validate_anndatas_sample_sizes(adatas):
        sample_sizes = [len(adata) for adata in adatas]
        if count_unique(sample_sizes) > 1:
            raise Exception("Please provide equal numbers of samples for all modalities.")

    
    @staticmethod
    def validate_anndatas_dimensions(adatas):
        for i, adata in enumerate(adatas):
            if len(adata.shape) != 2:
                raise Exception(f"The {i+1}-th modality in the given dataset is not 2D.")


    @staticmethod
    def validate_index_and_key(adatas, index, key):
        if (index is None and key is not None) or (index is not None and key is None):
            raise Exception(f"The provided label {label} and key {key} does not match.")
        if index is not None:
            DataManager.validate_index(adatas, index)
            DataManager.validate_key(adatas[index], key)


    @staticmethod
    def validate_index(adatas, index):
        if index is not None and index >= len(adatas):
            raise Exception(f"Provided reference modality index {index} is not valid.")


    @staticmethod
    def validate_key(adata, key):
        if key is not None and key not in adata.obs.keys():
            raise Exception(f"Provided reference modality key {key} is not valid.")


    @staticmethod 
    def get_Xs_from_anndatas(adatas):
        return [adata.X for adata in adatas]

    
    @staticmethod
    def get_obs_from_anndatas(adatas, index, key, default=None):
        if index is None:
            return default
        else:
            return adatas[index].obs[key]