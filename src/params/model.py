from dataclasses import dataclass,field
from typing import Literal,Optional,List,Union,Type

@dataclass
class ModelParams:
    model: str
    hidden_dim: int
    num_heads: int
    Resid: Optional[bool] = None
    ResidBias: Optional[bool] = None
    MessagePassing: Optional[bool] = False
    egatlayers: Optional[int] = 2
    Aggregate: Optional[str] = 'Concat'
    MixingLayer: Optional[bool] = False
    addons: Optional[bool] = None
    model_type: Optional[str] = 'default'
    targets: Optional[Union[str, List[str]]] = None
    dropout: Optional[float] = None
    NN_hidden_dim: Optional[Union[int, List[int]]] = 256
    NN_layers: Optional[int] = None
    cascading: Optional[bool] = False
    getattentionmaps: Optional[bool] = False
    getembeddings: Optional[int] = 0
    activation: Optional[Union[str, List[str]]] = None
    softmax: Optional[str] = None

class TuningParams:
    tuning: Optional[bool] = False
    tuning_method: Optional[str] = 'grid'
    tuning_iterations: Optional[int] = 10
    tuning_params: Optional[dict] = field(default_factory=dict)
    tuning_results: Optional[dict] = field(default_factory=dict)
    tuning_best_params: Optional[dict] = field(default_factory=dict)
    tuning_best_score: Optional[float] = None
    tuning_best_model: Optional[str] = None
    tuning_best_model_path: Optional[str] = None
    tuning_best_model_score: Optional[float] = None
    tuning_best_model_params: Optional[dict] = field(default_factory=dict)


@dataclass
class DenoteInputDataParams:
    denoteby: Optional[str] = None
    split: Optional[str] = None
    smilescol: Optional[str] = None
    fold: Optional[int] = None


class EGATModelParams(ModelParams,TuningParams,DenoteInputDataParams):
    pass