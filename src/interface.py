import anndata

from typing import List

from managers.data import Data, DataManager
from model import create_model_from_data, load_model_from_path, Model


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
        self.data = DataManager.format_anndatas(
            DataManager.train, adatas, batch_index, batch_key, label_index, label_key
        )
        self.model = create_model_from_data(self.data)


    def fit(self, 
        task: str,
        adatas_eval: List[anndata.AnnData] = None, 
        label_index_eval: int=None, 
        label_key_eval: str=None,
        batch_index_eval: int=None, 
        batch_key_eval: str=None,
    ):
        """\
        Fitting the current model on the saved training dataset.
        
        adatas_eval
            Used as evaluating dataset if provided. 
            Otherwise use the training dataset for evaluation.
        """
        self._check_data_exist()
             
        data_eval = self._format_data_or_retrieve_registered(
            DataManager.evaluate, adatas_eval, batch_index_eval, batch_key_eval, label_index_eval, label_key_eval
        )

        raise Exception("Not Implemented!")
        

    def evaluate(self, 
        adatas_eval: List[anndata.AnnData] = None, 
        label_index_eval: int=None, 
        label_key_eval: str=None,
        batch_index_eval: int=None, 
        batch_key_eval: str=None,
    ):
        """\
        Evaluate the current model with the adatas_eval dataset, or the registered data if the former not provided.

        adatas_eval
            The setting of adata_eval must match the saved adata.
        """
        self._check_model_exist()
        self._check_data_exist()

        data_eval = DataManager.format_anndatas(
            DataManager.evaluate, adatas_eval, batch_index_eval, batch_key_eval, label_index_eval, label_key_eval
        )

        raise Exception("Not Implemented!")


    def infer(self):
        """\
        Produce inference result for the adatas_infer dataset, or the registered data if the former not provided. 
        """
        self._check_model_exist()
        self._check_data_exist()

        raise Exception("Not Implemented!")


    def transfer(self,
        adatas_eval: List[anndata.AnnData] = None, 
        label_index_eval: int=None, 
        label_key_eval: str=None,
        batch_index_eval: int=None, 
        batch_key_eval: str=None,
    ):
        """\
        Perform ransfer learning on the adatas_transfer dataset. 

        adatas_transfer
            The setting of adata_infer must match the saved adata.
        """
        self._check_model_exist()
        self._check_data_exist()
             
        data_eval = self._format_data_or_retrieve_registered(
            DataManager.evaluate, adatas_eval, batch_index_eval, batch_key_eval, label_index_eval, label_key_eval
        )

        raise Exception("Not Implemented!")

    
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


    def _format_data_or_retrieve_registered(self, group, adatas, batch_index, batch_key, label_index, label_key):
        if adatas is None:
            return self.data
        else:
            return DataManager.format_anndatas(group, adatas, batch_index, batch_key, label_index, label_key)

    
    def _check_model_exist(self):
        if self.model is None:
            raise Exception("Please first train the model with the fit method or load model weights with load_model.")


    def _check_data_exist(self):
        if self.data is None:
            raise Exception("Please first register your training dataset with the register_anndatas method.")
