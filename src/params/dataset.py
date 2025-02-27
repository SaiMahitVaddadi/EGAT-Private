
from dataclasses import dataclass,field
from typing import Literal,Optional,List,Union,Type
import omegaconf

class DenotationParams:
    split_type: str
    splittotrain: str
    smilescolumn: str
    fold: Optional[int] = None

@dataclass
class DatasetParams:
    data_path: str
    rootfile: str
    exclude: Optional[List[str]] = None
    denotation: Optional[DenotationParams] = None
    splittotrain: Optional[str] = None
    smilescolumn: Optional[str] = None
    fold: Optional[int] = None
    target: Union[str, List[str]] = None
    additional: Optional[Union[str, List[str]]] = None
    graph: str = 'molecular'
    dimension: str = '2d'
    addmissingbonds: Optional[str] = None
    randomize: bool = False
    size: Optional[int] = None
    mode: str = 'dgl'
    jepa: bool = False
    jepa_masking: Optional[str] = None
    jepa_num_nodes: Optional[int] = None
    jepa_num_edges: Optional[int] = None
    jepa_neighbors: Optional[int] = None
    addons: Optional[Union[str, List[str]]] = None
    cache_size: int = 100
    n_splits: Optional[int] = None



@dataclass
class DataLoaderParams:
    test_only: bool = False
    root: str = ''
    class_choice: Optional[str] = None
    randomize: bool = False
    foldtype: Optional[str] = None
    size: Optional[int] = None
    batch_size: int = 32


class PredictParams:
    molecular: bool
    target: Union[str, List[str]]
    batch_size: int
    model_type: str
    additionals: Optional[Union[str, List[str]]] = None
    Norm: Optional[str] = None
    hasaddons: bool = False
    Embed: bool = False
    AttnMaps: bool = False
    tweights: Optional[List[float]] = None
    save_path: str


@dataclass
class NormalizerParams:
    normtarget: bool = False
    scaler: Type[str] = 'StandardScaler'
    scaler_model: Optional[str] = None
    root: str = ""
    model_type: str = "Hr"
    scaler_name: str = "scaler"


@dataclass
class SetupParams:
    data_path: str
    save_path: str
    model_path: Optional[str] = None
    base_model: Optional[str] = None
    ablation_EGAT_model: Optional[str] = None
    ablation_NN_model: Optional[str] = None
    loss: str = 'CrossEntropy'
    metric: str = 'Accuracy'
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    learning_rate_min: float = 1e-5
    momentum_orig: float = 0.9
    lr_decay: float = 0.1
    step_size: int = 10
    weight_decay: float = 1e-4
    epochs: int = 100
    gpu: int = 0
    parallel: bool = False
    setup: str = 'cuda'
    startpoint: str = 'Retrain'
    weightsandbiases: bool = False
    wandbproject: Optional[str] = None
    wandbname: Optional[str] = None
    model: str = 'EGAT'
    hasaddons: bool = False
    molecular: bool = False


@dataclass
class TrainParams:
    epoch: int
    epoch_const: int
    learning_rate: float
    lr_decay: float
    step_size: int
    expdecay: float
    warmup: int
    scheduler: str
    batch_size: int
    num_workers: int
    randomize: bool
    norm: Optional[str] = None
    scale_region: Optional[str] = None
    test_only: bool = False
    molecular: bool = False
    target: Union[str, List[str]] = 'target'
    tweights: List[float] = field(default_factory=list)
    model_type: str = 'direct'
    additionals: Optional[Union[str, List[str]]] = None
    hasaddons: bool = False
    Embed: bool = False
    AttnMaps: bool = False
    normtarget: bool = False
    trainedonnorm: bool = False
    metric: Optional[str] = None
    loss: Union[str, List[str]] = 'MAE'
    loss_agg: str = 'Arith'
    loss_weights: Optional[List[float]] = None
    patience: int = 10
    loss_threshold: float = 1e-4
    weightsandbiases: bool = False
    save_style: str = 'best'


class EGATParams(TrainParams, NormalizerParams, SetupParams, DatasetParams, DataLoaderParams, PredictParams):
    pass
    
    