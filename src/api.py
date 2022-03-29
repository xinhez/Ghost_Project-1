import anndata

from typing import List, Optional

from model import Model

from autofillers import autofill_batch, autofill_config
from validators import validate_anndatas

"""
API Interface
"""
class UnitedNet():
    """
    register_anndatas
        Save or override existing training dataset. If overriding existing training dataset, the model will also be refreshed.
            adatas: list of at least 2 modalities as anndata.
            reference_index_label: index of the label information modality.
            reference_index_batch: index of the batch information modality.
            obs_key_label: key to look up label information from the reference adata.
            obs_key_batch: key to look up batch information from the reference adata.
    """
    def register_anndatas(self, 
        adatas: List[anndata.AnnData], 
        reference_index_label: int=0,
        reference_index_batch: int=0,
        obs_key_label: str='label', 
        obs_key_batch: str='batch'
    ) -> None:
        validate_anndatas(adatas, reference_index_label, obs_key_label)

        self.modalities = [adata.X for adata in adatas]
        self.labels = adatas[reference_index_label].obs[obs_key_label]
        self.batches = autofill_batch(adatas[reference_index_batch], obs_key_batch, 0)

        self.model = Model(autofill_config(self.modalities, self.labels, self.batches))

    
    """
    load_model
        Load pretrained model parameters from the given path.
            path: absolute path to the desired location.
    """
    def load_model(self, path: str) -> None:
        pass

    
    """
    save_model
        Save current model parameters to the given path.
            path: absolute path to the desired location.
    """
    def save_model(self, path: str) -> None:
        pass


    """
    fit
        Fitting the current model on the saved training dataset.
            adata_eval: used as evaluating dataset if provided. Otherwise use the training dataset for evaluation.
    """
    def fit(self, adata_eval: anndata.AnnData=None):
        pass


    """
    evaluate
        Evaluate the current model on the provided dataset.
            adata_eval: The setting of adata_eval must match the saved adata.
    """
    def evaluate(self, adata_eval: anndata.AnnData):
        pass


    """
    infer
        Infer the current model on the provided dataset. 
            adata_infer: The setting of adata_infer must match the saved adata.
    """
    def infer(self, adata_infer: anndata.AnnData):
        pass