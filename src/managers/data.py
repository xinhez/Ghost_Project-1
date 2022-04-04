import numpy as np
import torch

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

from src.managers.base import NamedObject, ObjectManager
from src.utils import count_unique

class Dataset(TorchDataset):
    def __init__(self, modalities, batches, labels):
        super().__init__()
        self.modalities = [torch.Tensor(modality) for modality in modalities]
        self.batches = torch.LongTensor(batches)
        self.labels = None if labels is None else torch.LongTensor(labels)

    
    def __getitem__(self, index):
        modalities = [modality[index] for modality in self.modalities]
        if self.labels is None:
            return modalities, self.batches[index]
        else:
            return modalities, self.batches[index], self.labels[index]

    
    def __len__(self):
        return len(self.batches)


class Data(NamedObject):
    name = 'Data'

    def __init__(self, modalities, batches_or_batch, labels_or_None, *_):
        self.validate_batches(batches_or_batch)
        self.validate_labels(labels_or_None)

        self.modalities = modalities
        self.save_batches(batches_or_batch)
        self.labels = labels_or_None


    @property
    def input_sizes(self):
        return [modality.shape[1] for modality in self.modalities]

    
    @property
    def output_size(self):
        return count_unique(self.labels)

    
    @property
    def n_batch(self):
        return count_unique(self.batches)


    @property
    def n_sample(self):
        return len(self.modalities[0])


    def save_batches(self, batches_or_batch):
        if isinstance(batches_or_batch, list):
            self.batches = batches_or_batch
        else:
            self.batches = [batches_or_batch for _ in range(self.n_sample)]


    def validate_batches(self, batch_or_batches):
        if batch_or_batches is None:
            raise Exception("Unknow reason caused None batches.")


    def validate_labels(self, *_):
        pass            


    def create_dataset(self, model):
        batches = model.data_batch_encoder.fit_transform(self.batches)
        labels = None if self.labels is None else model.data_label_encoder.fit_transform(self.labels)
        return Dataset(self.modalities, batches, labels)


    def create_dataloader(self, model, batch_size):
        dataset = self.create_dataset(model)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class EvaluateData(Data):
    name = 'evaluate'


class InferData(Data):
    name = 'infer'


    @staticmethod
    def validate_modalities(modalities, modalities_provided, input_sizes):
        if len(modalities) != len(modalities_provided):
            raise Exception("Please check the claimed number of modalities information for data to be inferred.")

        for modality_index in modalities_provided:
            if modality_index >= len(input_sizes):
                raise Exception("The provided modality index exceeds the registered data modality number.")
                

    @staticmethod
    def autofill_modalities(modalities, modalities_provided, input_sizes):
        n_modality = len(input_sizes)

        modality_sizes = [(n_modality, input_size) for input_size in input_sizes]

        full_modalities = [np.zeros(modality_size) for modality_size in modality_sizes]

        for (i, modality) in zip(modalities_provided, modalities):
            full_modalities[i] = modality
        
        return full_modalities


    def __init__(self, modalities, batches_or_batch, labels_or_None, modalities_provided, input_sizes):
        InferData.validate_modalities(modalities, modalities_provided, input_sizes)

        modalities = InferData.autofill_modalities(modalities, modalities_provided, input_sizes)

        super().__init__(modalities, batches_or_batch, labels_or_None, modalities_provided, input_sizes)


class TrainData(Data):
    name = 'train'


    def validate_labels(self, labels_or_None):
        if labels_or_None is None:
            raise Exception("Training data must have non-None labels.")
        super().validate_labels(labels_or_None)


class TransferData(Data):
    name = 'transfer'


    def validate_labels(self, labels_or_None):
        if labels_or_None is None:
            raise Exception("Transfer data must have non-None labels.")
        super().validate_labels(labels_or_None)


class DataManager(ObjectManager):
    name = 'datas'
    constructors   = [EvaluateData, InferData, TrainData, TransferData]


    label = 'label'
    batch = 'batch'
    keys  = [label, batch]


    @staticmethod
    def format_anndatas(
        data_purpose, adatas, batch_index, batch_key, label_index=None, label_key=None, 
        modalities_provided=[], input_sizes=[], 
    ):
        DataManager.validate_anndatas(data_purpose, adatas, batch_index, batch_key, label_index, label_key)

        modalities       = DataManager.get_Xs_from_anndatas(adatas)
        batches_or_batch = DataManager.get_obs_from_anndatas(adatas, batch_index, batch_key, data_purpose)
        labels_or_None   = DataManager.get_obs_from_anndatas(adatas, label_index, label_key)
        
        constructor = DataManager.get_constructor_by_name(data_purpose)
        
        return constructor(modalities, batches_or_batch, labels_or_None, modalities_provided, input_sizes)


    @staticmethod
    def validate_anndatas(data_purpose, adatas, batch_index, batch_key, label_index, label_key):
        DataManager.validate_anndatas_exist(data_purpose, adatas)
        DataManager.validate_anndatas_sample_sizes(adatas)
        DataManager.validate_anndatas_dimensions(adatas)
        DataManager.validate_index_and_key(adatas, batch_index, batch_key)
        DataManager.validate_index_and_key(adatas, label_index, label_key)


    @staticmethod
    def validate_anndatas_exist(data_purpose, adatas):
        if adatas is None:
            raise Exception(f"Please provide AnnData for {data_purpose}.")


    @staticmethod
    def validate_anndatas_sample_sizes(adatas):
        sample_sizes = [len(adata) for adata in adatas]
        if count_unique(sample_sizes) > 1:
            raise Exception("Please provide equal numbers of samples for all modalities.")

    
    @staticmethod
    def validate_anndatas_dimensions(adatas):
        if len(adatas) == 0:
            raise Exception("Please provide at least 1 modality.")

        for i, adata in enumerate(adatas):
            if len(adata.shape) != 2:
                raise Exception(f"The {i+1}-th modality in the given dataset is not 2D.")


    @staticmethod
    def validate_index_and_key(adatas, index, key):
        if (index is None and key is not None) or (index is not None and key is None):
            raise Exception(f"The provided index {index} and key {key} must both or neither be None.")
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