import anndata
import numpy, random, torch
from regex import D

from typing import List

from src.config import ModelConfig, ScheduleConfig
from src.model import create_model_from_data, load_model_from_path, Model
from src.managers.data import Data, DataManager
from src.managers.data import EvaluationData, InferenceData, TrainingData, TransferenceData, ValidationData
from src.managers.task import CustomizedTask, TaskManager
from src.utils import set_random_seed


set_random_seed(numpy, random, torch)


class UnitedNet:
    """\
    API Interface
    """
    def register_anndatas(self, 
        adatas: List[anndata.AnnData], 
        label_index: int,
        label_key: str, 
        batch_index: int=None,
        batch_key: str=None,
    ) -> None:
        """\
        Save or override existing training dataset. 
        If overriding existing training dataset, the model will also be refreshed.

        adatas
            list of at modalities as anndata.
        reference_index_batch
            index of the modality to look up batch information.
        obs_key_batch
            key to look up batch information from the reference adata.
        reference_index_label
            index of the modality to look up label information.
        obs_key_label
            key to look up label information from the reference adata.
        """
        self.data = DataManager.format_anndatas(
            TrainingData.name, adatas, batch_index, batch_key, label_index, label_key
        )
        self.model = create_model_from_data(self.data)


    def fit(self, 
        task:                 str,
        adatas_validate:      List[anndata.AnnData] = None, 
        label_index_validate: int                   = None, 
        label_key_validate:   str                   = None,
        batch_index_validate: int                   = None, 
        batch_key_validate:   str                   = None,
        n_epoch:              int                   = 1,
        batch_size:           int                   = 512,
        schedule_configs:     List[ScheduleConfig]  = None,
        save_log_path:        str                   = None,
        device:               str                   = 'cpu',
    ):
        """\
        Fitting the current model on the saved training dataset.
        
        adatas_eval
            Used as evaluating dataset if provided. 
            Otherwise use the training dataset for evaluation.
        """
        self._check_model_exist()
        self.set_device(device)

        self._check_data_exist()
             
        data_validation = self._format_data_or_retrieve_registered(
            ValidationData.name, adatas_validate, 
            batch_index_validate, batch_key_validate, 
            label_index_validate, label_key_validate
        )

        task_manager = TaskManager.get_constructor_by_name(task)()
        task_manager.train(schedule_configs, self.model, self.data, data_validation, batch_size, n_epoch, save_log_path)


    def transfer(self,
        task:                     str,
        adatas_transference:      List[anndata.AnnData], 
        label_index_transference: int, 
        label_key_transference:   str,
        batch_index_transference: int                   = None, 
        batch_key_transference:   str                   = None,
        adatas_validation:        List[anndata.AnnData] = None, 
        label_index_validation:   int                   = None, 
        label_key_validation:     str                   = None,
        batch_index_validation:   int                   = None, 
        batch_key_validation:     str                   = None,
        n_epoch:                  int                   = 1,
        batch_size:               int                   = 512,
        schedule_configs:         List[ScheduleConfig]  = None,
        save_log_path:            str                   = None,
        device:                   str                   = 'cpu',
    ):
        """\
        Perform ransfer learning on the adatas_transfer dataset. 

        adatas_transfer
            The setting of adata_infer must match the saved adata.
        """
        self._check_model_exist()
        self.set_device(device)

        self._check_data_exist()

        data_transfer = DataManager.format_anndatas(
            TransferenceData.name, adatas_transference, 
            batch_index_transference, batch_key_transference, 
            label_index_transference, label_key_transference
        )
             
        data_validation = self._format_data_or_retrieve_registered(
            ValidationData.name, adatas_validation, 
            batch_index_validation, batch_key_validation, 
            label_index_validation, label_key_validation
        )

        task_manager = TaskManager.get_constructor_by_name(task)()
        task_manager.transfer(
            schedule_configs, self.model, self.data, data_transfer, data_validation, batch_size, n_epoch, save_log_path
        )


    def evaluate(self, 
        adatas_evaluation:      List[anndata.AnnData] = None, 
        label_index_evaluation: int                   = None, 
        label_key_evaluation:   str                   = None,
        batch_index_evaluation: int                   = None, 
        batch_key_evaluation:   str                   = None,
        batch_size:             int                   = 512,
        save_log_path:          str                   = None,
        device:                 str                   = 'cpu',
    ):
        """\
        Evaluate the current model with the adatas_eval dataset, or the registered data if the former not provided.

        adatas_eval
            The setting of adata_eval must match the saved adata.
        """
        self._check_model_exist()
        self.set_device(device)

        data_evaluation = self._format_data_or_retrieve_registered(
            EvaluationData.name, adatas_evaluation, 
            batch_index_evaluation, batch_key_evaluation, 
            label_index_evaluation, label_key_evaluation
        )

        task_manager = TaskManager.get_constructor_by_name(CustomizedTask.name)()
        return task_manager.evaluate(self.model, data_evaluation, batch_size, save_log_path)


    def infer(self,
        adatas_inference:        List[anndata.AnnData] = None, 
        modalities_provided:     List                  = [],
        batch_index_inference:   int                   = None, 
        batch_key_inference:     str                   = None,
        batch_size:              int                   = 512,
        save_log_path:           str                   = None,
        modality_sizes:          List[int]             = None,
        device:                  str                   = 'cpu',
    ):
        """\
        Produce inference result for the adatas_infer dataset, or the registered data if the former not provided. 
        """
        self._check_model_exist()
        self.set_device(device)
        
        if modality_sizes is None:
            self._check_data_exist()

        data_inference = self._format_data_or_retrieve_registered(
            InferenceData.name, adatas_inference, batch_index_inference, batch_key_inference, 
            modalities_provided=modalities_provided, modality_sizes=modality_sizes or self.data.modality_sizes,
        )

        task_manager = TaskManager.get_constructor_by_name(CustomizedTask.name)()
        return task_manager.infer(self.model, data_inference, batch_size, save_log_path, modalities_provided)


    def load_model(self, path: str) -> None:
        """\
        Load pretrained model parameters from the given path.
        
        path
            The absolute path to the desired location.
        """
        self.model = load_model_from_path(path)


    def save_model(self, path: str) -> None:
        """\
        Save current model parameters to the given path.
        path
            The absolute path to the desired location.
        """
        self.model.save_model(path)


    def update_model_config(self, config: ModelConfig) -> None:
        """\
        Update the model configurations.
        """
        self._check_model_exist()

        self.model.update_config(config)

    
    def set_device(self, device: str='cpu'):
        self.model.save_device_in_use(device)
        self.model = self.model.to(device=device)


    def _get_data(self):
        return getattr(self, Data.name)

    def _set_data(self, data):
        setattr(self, Data.name, data)

    data = property(_get_data, _set_data)


    def _get_model(self):
        return getattr(self, Model.name)

    def _set_model(self, model):
        setattr(self, Model.name, model)

    model = property(_get_model, _set_model)


    def _format_data_or_retrieve_registered(self, 
        group, adatas, batch_index, batch_key, 
        label_index=None, label_key=None,
        modalities_provided=[], modality_sizes=[],
    ):
        if adatas is None:
            return self.data
        else:
            return DataManager.format_anndatas(
                group, adatas, batch_index, batch_key, label_index, label_key, modalities_provided, modality_sizes,
            )


    def _check_model_exist(self):
        if self.model is None:
            raise Exception(
                "Please first generate model config with register_anndata or load model weights with load_model."
            )


    def _check_data_exist(self):
        if self.data is None:
            raise Exception("Please first register your training dataset with the register_anndatas method.")