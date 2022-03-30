import anndata

from typing import List

from model import create_model_from_data, load_model_from_path


class UnitedNet():
    """\
    API Interface
    """
    def register_anndatas(self, 
        adatas: List[anndata.AnnData], 
        obs_key_label: str='label', 
        obs_key_batch: str='batch',
        reference_index_label: int=0,
        reference_index_batch: int=0,
    ) -> None:
        """\
        Save or override existing training dataset. If overriding existing training dataset, the model will also be refreshed.

        adatas
            list of at least 2 modalities as anndata.
        obs_key_label
            key to look up label information from the reference adata.
        obs_key_batch
            key to look up batch information from the reference adata.
        reference_index_label
            index of the label information modality.
        reference_index_batch
            index of the batch information modality.
        """
        self.model = create_model_from_data(
            adatas, 
            reference_index_label, reference_index_batch, 
            obs_key_label, obs_key_batch
        )
        self.adatas = adatas

    
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


    def fit(self, adata_eval: anndata.AnnData=None):
        """\
        Fitting the current model on the saved training dataset.
        
        adata_eval
            Used as evaluating dataset if provided. Otherwise use the training dataset for evaluation.
        """
        pass


    def evaluate(self, adata_eval: anndata.AnnData):
        """\
        Evaluate the current model on the provided dataset.

        adata_eval
            The setting of adata_eval must match the saved adata.
        """
        pass


    def infer(self, adata_infer: anndata.AnnData):
        """\
        Infer the current model on the provided dataset. 

        adata_infer
            The setting of adata_infer must match the saved adata.
        """
        pass