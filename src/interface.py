import anndata
import numpy
import random
import torch
import tensorflow as tf

from typing import List, Union

from src.config import ModelConfig, TaskConfig, TechniqueConfig
from src.logger import Logger
from src.model import load_model_from_path, Model
from src.managers.data import Data, DataManager
from src.managers.data import (
    EvaluationData,
    InferenceData,
    TrainingData,
    TransferenceData,
    ValidationData,
)
from src.managers.task import BaseTask, TaskManager
from src.managers.technique import DefaultTechnique, TechniqueManager
from src.utils import set_random_seed


class UnitedNet:
    """\
    API Interface
    """

    def __init__(
        self,
        device="cpu",
        log_path="saved_log",
        model_path="saved_models",
        tensorboard_path="saved_tensorboards",
        verbose=True,
        random_seed=3407,
    ):
        self.set_random_seed(random_seed)
        self.set_device(device)
        self.set_verbose(verbose)
        self.set_log_path(log_path)
        self.set_model_path(model_path)
        self.set_tensorboard_path(tensorboard_path)

    def register_anndatas(
        self,
        adatas: List[anndata.AnnData],
        label_index: int,
        label_key: str,
        batch_index: int = None,
        batch_key: str = None,
        technique: str = DefaultTechnique.name,
    ) -> None:
        """\
        Save or override existing training dataset. 
        If overriding existing training dataset, the model will also be refreshed.
        """
        self.data = DataManager.format_anndatas(
            TrainingData.name,
            adatas,
            batch_index,
            batch_key,
            label_index,
            label_key,
        )
        self.technique = TechniqueManager.get_constructor_by_name(technique)(self.data)
        self.model = Model(self.technique.get_model_config())
        self.set_model_device()

    def train(
        self,
        task: Union[str, TaskConfig],
        adatas_validate: List[anndata.AnnData] = None,
        label_index_validate: int = None,
        label_key_validate: str = None,
        batch_index_validate: int = None,
        batch_key_validate: str = None,
        n_epoch: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        save_best_model: bool = False,
        checkpoint: int = 0,
    ):
        """\
        Training the current model on the saved training dataset.
        """
        data_validate = self._retrieve_validation_data(
            adatas_validate,
            label_index_validate,
            label_key_validate,
            batch_index_validate,
            batch_key_validate,
        )

        self._start_writer()

        TaskManager.get_task_by_name_or_config(task).train(
            task,
            self.model,
            self.data,
            data_validate,
            batch_size,
            n_epoch,
            learning_rate,
            self.logger,
            self.model_path,
            save_best_model,
            checkpoint,
            self.writer,
            self.random_seed,
        )

        self._close_writer()

    def finetune(
        self,
        task: Union[str, TaskConfig],
        adatas_validate: List[anndata.AnnData] = None,
        label_index_validate: int = None,
        label_key_validate: str = None,
        batch_index_validate: int = None,
        batch_key_validate: str = None,
        n_epoch: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        save_best_model: bool = False,
        checkpoint: int = 0,
    ):
        """\
        Training the current model on the saved training dataset.
        """
        data_validate = self._retrieve_validation_data(
            adatas_validate,
            label_index_validate,
            label_key_validate,
            batch_index_validate,
            batch_key_validate,
        )

        self._start_writer()

        TaskManager.get_task_by_name_or_config(task).finetune(
            task,
            self.model,
            self.data,
            data_validate,
            batch_size,
            n_epoch,
            learning_rate,
            self.logger,
            self.model_path,
            save_best_model,
            checkpoint,
            self.writer,
            self.random_seed,
        )

        self._close_writer()

    def transfer(
        self,
        task: Union[str, TaskConfig],
        adatas_transfer: List[anndata.AnnData],
        batch_index_transfer: int = None,
        batch_key_transfer: str = None,
        adatas_validate: List[anndata.AnnData] = None,
        label_index_validate: int = None,
        label_key_validate: str = None,
        batch_index_validate: int = None,
        batch_key_validate: str = None,
        n_epoch: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        save_best_model: bool = False,
        checkpoint: int = 0,
    ):
        """\
        Perform ransfer learning on the adatas_transfer dataset. 
        """
        data_validate = self._retrieve_validation_data(
            adatas_validate,
            label_index_validate,
            label_key_validate,
            batch_index_validate,
            batch_key_validate,
        )

        data_transfer = DataManager.format_anndatas(
            TransferenceData.name,
            adatas_transfer,
            batch_index_transfer,
            batch_key_transfer,
        )

        self._start_writer()

        TaskManager.get_task_by_name_or_config(task).transfer(
            task,
            self.model,
            self.data,
            data_transfer,
            data_validate,
            batch_size,
            n_epoch,
            learning_rate,
            self.logger,
            self.model_path,
            save_best_model,
            checkpoint,
            self.writer,
            self.random_seed,
        )

        self._close_writer()

    def evaluate(
        self,
        adatas_evaluate: List[anndata.AnnData] = None,
        label_index_evaluate: int = None,
        label_key_evaluate: str = None,
        batch_index_evaluate: int = None,
        batch_key_evaluate: str = None,
    ):
        """\
        Evaluate the current model with the adatas_eval dataset, or the registered data if the former not provided.
        """
        self._check_model_exist()

        data_evaluation = self._format_data_or_retrieve_registered(
            EvaluationData.name,
            adatas_evaluate,
            batch_index_evaluate,
            batch_key_evaluate,
            label_index_evaluate,
            label_key_evaluate,
        )

        return BaseTask().evaluate(self.model, data_evaluation, self.logger)

    def infer(
        self,
        adatas_infer: List[anndata.AnnData] = None,
        modalities_provided: List = [],
        batch_index_infer: int = None,
        batch_key_infer: str = None,
        modality_sizes: List[int] = None,
    ) -> Union[List[List[anndata.AnnData]], anndata.AnnData]:
        """\
        Produce inference result for the adatas_infer dataset, or the registered data if the former not provided. 
        """
        self._check_model_exist()

        if modality_sizes is None:
            self._check_data_exist()

        data_inference = self._format_data_or_retrieve_registered(
            InferenceData.name,
            adatas_infer,
            batch_index_infer,
            batch_key_infer,
            modalities_provided=modalities_provided,
            modality_sizes=modality_sizes or self.data.modality_sizes,
        )

        return BaseTask().infer(
            self.model, data_inference, self.logger, modalities_provided
        )

    def load_model(self, path: str) -> None:
        """\
        Create a new model with all the model parameters from the given path (usually from checkpoint).

        path
            The absolute path to the desired location.
        """
        self.model = load_model_from_path(path)
        self.set_model_device()

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
        self.set_model_device()

    def update_technique_config(self, config: TechniqueConfig) -> None:
        self.technique.update_config(config)
        self.model = Model(self.technique.get_model_config())
        self.set_model_device()

    def set_device(self, device: str = "cpu"):
        self.device = device

    def set_model_device(self):
        self._check_model_exist()
        self.model = self.model.to(device=self.device)
        self.model.set_device_in_use(self.device)

    def set_log_path(self, log_path):
        self.logger = Logger(log_path, self.verbose)

    def set_model_path(self, model_path):
        self.model_path = model_path

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
        set_random_seed(numpy, random, torch, self.random_seed)

    def set_tensorboard_path(self, tensorboard_path):
        self.tensorboard_path = tensorboard_path

    def set_verbose(self, verbose):
        self.verbose = verbose

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

    def _format_data_or_retrieve_registered(
        self,
        group,
        adatas,
        batch_index,
        batch_key,
        label_index=None,
        label_key=None,
        modalities_provided=[],
        modality_sizes=[],
    ):
        if adatas is None:
            return self.data
        else:
            return DataManager.format_anndatas(
                group,
                adatas,
                batch_index,
                batch_key,
                label_index,
                label_key,
                modalities_provided,
                modality_sizes,
            )

    def _check_model_exist(self):
        if self.model is None:
            raise Exception(
                "Please first generate model config with register_anndata or load model weights with load_model."
            )

    def _check_data_exist(self):
        if self.data is None:
            raise Exception(
                "Please first register your training dataset with the register_anndatas method."
            )

    def _retrieve_validation_data(
        self,
        adatas_validate,
        label_index_validate,
        label_key_validate,
        batch_index_validate,
        batch_key_validate,
    ):
        self._check_model_exist()
        self._check_data_exist()
        return self._format_data_or_retrieve_registered(
            ValidationData.name,
            adatas_validate,
            batch_index_validate,
            batch_key_validate,
            label_index_validate,
            label_key_validate,
        )

    def _start_writer(self):
        if self.tensorboard_path is not None:
            self.writer = tf.summary.create_file_writer(self.tensorboard_path)
        else:
            self.writer = None

    def _close_writer(self):
        if self.writer is not None:
            self.writer.close()
