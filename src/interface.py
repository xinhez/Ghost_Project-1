import anndata

from typing import List

from constants import BATCH, LABEL, EVALUATION, INFERENCE, MODALITIES, MODEL
from dataset import format_data, validate_data_sizes_match
from model import create_model_from_data, load_model_from_path
from scheduler import evaluate, infer, train


class UnitedNet():
    """\
    API Interface
    """
    def register_anndatas(self, 
        adatas: List[anndata.AnnData], 
        reference_index_batch: int=0,
        obs_key_batch: str=BATCH,
        reference_index_label: int=0,
        obs_key_label: str=LABEL, 
    ) -> None:
        """\
        Save or override existing training dataset. 
        If overriding existing training dataset, the model will also be refreshed.

        adatas
            list of at least 2 modalities as anndata.
        reference_index_batch
            index of the modality to look up batch information.
        obs_key_batch
            key to look up batch information from the reference adata.
        reference_index_label
            index of the modality to look up label information.
        obs_key_label
            key to look up label information from the reference adata.
        """
        self.data = format_data(
            adatas, 
            reference_index_label, obs_key_label, 
            reference_index_batch, obs_key_batch,
        )
        self.model = create_model_from_data(self.data)

    
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


    def fit(self, 
        task: str,
        adatas_eval: List[anndata.AnnData] = None, 
        reference_index_batch: int=0, 
        obs_key_batch: str=BATCH,
        reference_index_label: int=0, 
        obs_key_label: str=LABEL, 
    ):
        """\
        Fitting the current model on the saved training dataset.
        
        adatas_eval
            Used as evaluating dataset if provided. 
            Otherwise use the training dataset for evaluation.
        """
        data_eval = self._check_and_process_evaluation_data(
            adatas_eval, 
            reference_index_batch, obs_key_batch,
            reference_index_label, obs_key_label, 
        )
        train(self.model, task, self.data, data_eval)


    def evaluate(self, 
        adatas_eval: List[anndata.AnnData] = None, 
        reference_index_batch: int=0, 
        obs_key_batch: str=BATCH,
        reference_index_label: int=0, 
        obs_key_label: str=LABEL, 
    ):
        """\
        Evaluate the current model on the provided dataset.

        adata_eval
            The setting of adata_eval must match the saved adata.
        """
        self._check_model_exist()
        data_eval = self._check_and_process_evaluation_data(
            adatas_eval, 
            reference_index_batch, obs_key_batch,
            reference_index_label, obs_key_label, 
        )
        evaluate(self.model, data_eval)


    def infer(self, 
        adatas_eval: List[anndata.AnnData] = None, 
        reference_index_batch: int=0,
        obs_key_batch: str=BATCH,
    ):
        """\
        Infer the current model on the provided dataset. 

        adata_infer
            The setting of adata_infer must match the saved adata.
        """
        self._check_model_exist()
        data_eval = self._check_and_process_evaluation_data(
            adatas_eval, reference_index_batch, obs_key_batch,
        )
        infer(self.model, data_eval)

    
    def _check_and_process_evaluation_data(self, 
        adatas_eval, 
        reference_index_batch, obs_key_batch,
        reference_index_label=None, obs_key_label=None, 
    ):
        if not hasattr(self, MODALITIES):
            raise Exception("Please first register your training dataset with the register_anndatas method.")

        if adatas_eval is not None:
            data_eval = format_data(
                adatas_eval, 
                reference_index_batch, obs_key_batch,
                reference_index_label, obs_key_label,
                default_batch = INFERENCE if reference_index_label is None else EVALUATION,
            )
            validate_data_sizes_match(self.data, data_eval)
        else: 
            data_eval = self.data

        return data_eval

    
    def _check_model_exist(self):
        if not hasattr(self, MODEL):
            raise Exception("Please first train the model with the fit method or load model weights with load_model.")