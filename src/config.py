from pydantic import BaseModel
from typing import List, Literal, NewType, Union

NoneStrList = NewType('NoneStrList', List[Union[None, str]])
NoneFloatList = NewType('NoneFloatList', List[Union[None, float]])
BoolOrBoolList = NewType('BoolOrBoolList', Union[bool, List[bool]])


class Config(BaseModel):
    @property
    def class_name(self):
        return self.__class__.__name__


class MLP(Config):
    input_size: int
    output_size: int
    hidden_sizes: List[int] = [64, 64]
    biases: BoolOrBoolList
    dropouts: Union[float, None, NoneFloatList]
    batchnorms: BoolOrBoolList
    activations: Union[str, None, NoneStrList]


class Model(Config):
    encoders: List[MLP]
    decoders: List[MLP]
    fusion_method: Literal["mean", "weighted_mean", "weighted_mean_feature", "attention"]
    


"""
create_default_config
"""
def create_default_config(input_sizes, output_size, n_batch):
    return Model