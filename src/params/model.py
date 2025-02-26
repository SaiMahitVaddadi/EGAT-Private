
from dataclasses import dataclass,field
from typing import Literal,Optional,List,Union,Type
import omegaconf
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

@dataclass
class ModelParams:
    # MandatoryModelArgs
    model: str = 'EGAT_1MLP_NOUT'
    model_type: Literal['direct', 'multi', 'Hr','Hr_multi','Hr2_multi','BEP'] = 'direct'
    data_path: str = ''
    rootfile: str = ''
    batch_size: int = 32
    additionals: Optional[Union[List[str], omegaconf.listconfig.ListConfig]] = None
    targets: Optional[Union[List[str], omegaconf.listconfig.ListConfig]] = None
    hasaddons: bool = False
    scaler: Literal['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'Normalizer',None] = None
    scaler_model: Optional[str] = None
    root: str = './'
    normtarget: bool = False
    exclude: Optional[Union[str, List[str]]] = None
    test_only: bool = False
    class_choice: Optional[str] = None
    randomize: bool = True
    fold: Optional[int] = None
    foldtype: Optional[str] = None
    size: Optional[int] = None
    addons: Optional[Union[str, List[str]]] = None
    weightsandbiases: bool = False
    wandbproject: str = ''
    wandbname: str = ''
    setup: str = 'cpu'
    parallel: bool = False
    save_path: str = ''
    startpoint: str = 'Retrain'

    # OptionalModelParams
    EGAT_layers: int = 4
    Resid: Optional[bool] = None
    ResidBias: Optional[bool] = None
    MessagePassing: Optional[bool] = None
    num_heads: int = 4
    hidden_dim: int = 128

    NN_layers: int = 3
    NN_hidden_dim: Union[int, List[int]] = 128

    dropout: Union[float, List[float]] = None
    activation: Optional[Union[str, List[str]]] = None
    softmax: Optional[str] = None
    Aggregate: Optional[str] = 'Concat'
    MixingLayer: Optional[bool] = None
    graph: Optional[str] = 'molecular'
    SA: Optional[bool] = None

    dimension: str = '2d'
    addmissingbonds: Optional[str] = None
    jepa: bool = False
    jepa_masking: Optional[str]

    dimension: str = '2d'
    addmissingbonds: Optional[str] = None
    jepa: bool = False
    jepa_masking: Optional[str] = None
    jepa_num_nodes: Optional[int] = None
    jepa_num_edges: Optional[int] = None
    jepa_neighbors: Optional[int] = None
    cache_size: int = 1000
    randomize: bool = False
    size: Optional[int] = None
    split_type: Optional[str] = None
    splittotrain: Optional[str] = None
    smilescolumn: Optional[str] = None
    fold: Optional[int] = None
    n_splits: Optional[int] = None
    mode: str = 'dgl'
    dataset: Optional[str] = None


@dataclass
class BaseParams:
    base_model: Optional[str] = None
    ablation_EGAT_model: Optional[str] = None
    ablation_NN_model: Optional[str] = None
    loss: Union[str, List[str]] = 'MAE'
    data_path: str = ''
    hasaddons: bool = False
    molecular: bool = False
    defaults: Optional[dict] = None
    learning_rate_min: float = 0.0
    momentum_orig: float = 0.9
    lr_decay: float = 0.1
    step_size: int = 10
    epochs: int = 100
    epochs_const: int = 0
    scheduler: str = 'cosine'
    gamma: float = 0.1
    optimizer: str = 'adam'
    lr: float = 0.001
    weight_decay: float = 0.0


@dataclass
class TrainParams:
    epoch: int = 100
    epoch_const: int = 0
    scheduler: str = 'step'
    learning_rate: float = 0.001
    lr_decay: float = 0.1
    step_size: int = 10
    warmup: int = 5
    expdecay: float = 0.1
    molecular: bool = False
    target: Union[str, List[str]] = 'target'
    additionals: Optional[Union[str, List[str]]] = None
    model_type: str = 'direct'
    Norm: Optional[str] = None
    Embed: bool = False
    AttnMaps: bool = False
    tweights: Optional[List[float]] = None
    hasaddons: bool = False
    test_only: bool = False
    weightsandbiases: bool = False
    patience: int = 10
    loss_threshold: float = 0.01
    batch_size: int = 32
    randomize: bool = True
    num_workers: int = 4
    scale_region: str = 'all'
    normtarget: bool = False
    trainedonnorm: bool = False
    metric: Optional[str] = None
    metrics: Optional[Union[str, List[str]]] = None
    loss_agg: str = 'Arith'
    loss_weights: Optional[List[float]] = None
    save_style: str = 'best'






@dataclass
class EGATParams(ModelParams, BaseParams, TrainParams):
    pass


