import anndata

from typing import List

from src.config import ModelConfig, ScheduleConfig
from src.model import create_model_from_data, load_model_from_path, Model
from src.managers.data import Data, DataManager, EvaluateData, InferData, TrainData, TransferData
from src.managers.task import CustomizedTask, TaskManager


class UnitedNet():
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
            TrainData.name, adatas, batch_index, batch_key, label_index, label_key
        )
        self.model = create_model_from_data(self.data)


    def fit(self, 
        task:             str                   = CustomizedTask.name,
        adatas_eval:      List[anndata.AnnData] = None, 
        label_index_eval: int                   = None, 
        label_key_eval:   str                   = None,
        batch_index_eval: int                   = None, 
        batch_key_eval:   str                   = None,
        n_epoch:          int                   = 1,
        batch_size:       int                   = 512,
        schedule_configs: List[ScheduleConfig]        = None,
        save_log_path:    str                   = None,
        device:           str                   = 'cpu',
    ):
        """\
        Fitting the current model on the saved training dataset.
        
        adatas_eval
            Used as evaluating dataset if provided. 
            Otherwise use the training dataset for evaluation.
        """
        self._check_model_exist()
        self.model.set_device(device)

        self._check_data_exist()
             
        data_eval = self._format_data_or_retrieve_registered(
            EvaluateData.name, adatas_eval, batch_index_eval, batch_key_eval, label_index_eval, label_key_eval
        )

        task_manager = TaskManager.get_constructor_by_name(task)()
        task_manager.train(self.model, self.data, batch_size, n_epoch, schedule_configs, save_log_path)
        return task_manager.evaluate(self.model, data_eval, batch_size, save_log_path)


    def transfer(self,
        adatas_transfer:      List[anndata.AnnData], 
        label_index_transfer: int, 
        label_key_transfer:   str,
        task:                 str                   = CustomizedTask.name,
        batch_index_transfer: int                   = None, 
        batch_key_transfer:   str                   = None,
        adatas_eval:          List[anndata.AnnData] = None, 
        label_index_eval:     int                   = None, 
        label_key_eval:       str                   = None,
        batch_index_eval:     int                   = None, 
        batch_key_eval:       str                   = None,
        n_epoch:              int                   = 1,
        batch_size:           int                   = 512,
        schedule_configs:     List[ScheduleConfig]        = None,
        save_log_path:        str                   = None,
        device:               str                   = 'cpu',
    ):
        """\
        Perform ransfer learning on the adatas_transfer dataset. 

        adatas_transfer
            The setting of adata_infer must match the saved adata.
        """
        self._check_model_exist()
        self.model.set_device(device)

        self._check_data_exist()

        data_transfer = DataManager.format_anndatas(
            TransferData.name, 
            adatas_transfer, batch_index_transfer, batch_key_transfer, label_index_transfer, label_key_transfer
        )
             
        data_eval = self._format_data_or_retrieve_registered(
            EvaluateData.name, adatas_eval, batch_index_eval, batch_key_eval, label_index_eval, label_key_eval
        )

        task_manager = TaskManager.get_constructor_by_name(task)()
        task_manager.transfer(
            self.model, self.data, data_transfer, batch_size, n_epoch, schedule_configs, save_log_path
        )
        task_manager.evaluate(self.model, data_eval, batch_size, save_log_path)


    def evaluate(self, 
        adatas_eval:      List[anndata.AnnData] = None, 
        label_index_eval: int                   = None, 
        label_key_eval:   str                   = None,
        batch_index_eval: int                   = None, 
        batch_key_eval:   str                   = None,
        batch_size:       int                   = 512,
        save_log_path:    str                   = None,
        device:           str                   = 'cpu',
    ):
        """\
        Evaluate the current model with the adatas_eval dataset, or the registered data if the former not provided.

        adatas_eval
            The setting of adata_eval must match the saved adata.
        """
        self._check_model_exist()
        self.model.set_device(device)

        data_eval = self._format_data_or_retrieve_registered(
            EvaluateData.name, adatas_eval, batch_index_eval, batch_key_eval, label_index_eval, label_key_eval
        )

        task_manager = TaskManager.get_constructor_by_name(CustomizedTask.name)()
        return task_manager.evaluate(self.model, data_eval, batch_size, save_log_path)


    def infer(self,
        adatas_infer:        List[anndata.AnnData] = None, 
        modalities_provided: List                  = [],
        batch_index_infer:   int                   = None, 
        batch_key_infer:     str                   = None,
        batch_size:          int                   = 512,
        save_log_path:       str                   = None,
        modality_sizes:      List[int]             = None,
        device:              str                   = 'cpu',
    ):
        """\
        Produce inference result for the adatas_infer dataset, or the registered data if the former not provided. 
        """
        self._check_model_exist()
        self.model.set_device(device)
        
        if modality_sizes is None:
            self._check_data_exist()

        data_infer = self._format_data_or_retrieve_registered(
            InferData.name, adatas_infer, batch_index_infer, batch_key_infer, 
            modalities_provided=modalities_provided, modality_sizes=modality_sizes or self.data.modality_sizes,
        )

        task_manager = TaskManager.get_constructor_by_name(CustomizedTask.name)()
        return task_manager.infer(self.model, data_infer, batch_size, save_log_path, modalities_provided)


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
            raise Exception("Please first generate model config with register_anndata or load model weights with load_model.")


    def _check_data_exist(self):
        if self.data is None:
            raise Exception("Please first register your training dataset with the register_anndatas method.")