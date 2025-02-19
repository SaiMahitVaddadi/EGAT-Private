
from dataclasses import dataclass,field
from typing import Literal,Optional,List,Union


@dataclass
class ModelParams:
    model: str = 'EGAT_1MLP_NOUT'
    EGAT_layers: int = 4
    Resid: Optional[bool] = None
    ResidBias: Optional[bool] = None
    MessagePassing: Optional[bool] = None
    num_heads: int = 4
    hidden_dim: int = 128

    NN_layers: int = 3
    NN_hidden_dim: Union[int, List[int]] = 128

    dropout: Union[float, List[float]] = 0.5
    activation: Optional[Union[str, List[str]]] = None
    softmax: Optional[str] = None
    Aggregate: Optional[str] = 'Concat'
    MixingLayer: Optional[bool] = None
    addons: Optional[bool] = None
    model_type: Optional[str] = 'direct'
    targets: Optional[Union[str, List[str]]] = None
    graph: Optional[str] = 'molecular'
    SA: Optional[bool] = None
    data_path: str
    rootfile: str
    exclude: Optional[List[str]] = None
    target: Optional[Union[str, List[str]]] = None
    additional: Optional[Union[str, List[str]]] = None

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
class GraphParams:
    neighbor: str = 'all'
    removeringinfo: bool = False
    removearomaticity: bool = False
    removehybridinfo: bool = False
    useFullHyb: bool = False
    removeformalchargeinfo: bool = False
    charge: str = 'Gasteiger'
    getacidbaseinfo: bool = False
    getspiro: bool = False
    getbridgehead: bool = False
    getrotatablebonds: bool = False
    removechiralinfo: bool = False
    removeconjinfo: bool = False
    removestereoinfo: bool = False
    getbondlength: bool = False
    getatomdistance: bool = False
    getbondangle: str = 'first'
    getdihedral: str = 'first'
    getbondmidpoint: bool = False
    getbrics: bool = False
    checkbricsbond: bool = False
    removecoordinationinfo: bool = False
    getdistancetocenterofmass: bool = False
    getsterichindrance: bool = False
    getasa: bool = False
    getgaussiancurvature: bool = False
    getmolecularshapeindex: bool = False
    getdistancetoconvexhull: bool = False
    removereactiveinfo: bool = False
    addneighboringreactives: bool = False
    gethybridizationchange: bool = False
    getdegreecentrality: bool = False
    getvalencychange: bool = False
    getoxidationreduction: bool = False
    getbondordersumchange: bool = False
    getneighborhoodchangeratio: bool = False
    getLocalAtomicEnvironmentSimilarity: bool = False
    adddisttoreactingbonds: bool = False
    addmissingbonds: str = 'hbonds'
    addcho: bool = False
    adddihydrogenbonds: bool = False
    addcationpi: bool = False
    addpipistack: bool = False
    addhalogenbonds: bool = False
    addmetallophilic: bool = False
    addelectrostatic: bool = False
    addeneg: bool = False
    getglobal: bool = False
    getshortestpath: bool = False
    getshortestpathweighted: bool = False
    getrandomwalk: bool = False
    getcommutetimes: bool = False
    getshortestpathcount: bool = False
    geteffectiveresistance: bool = False
    getcommonneighbors: bool = False
    getpctcommonneighbors: bool = False
    getglobaljaccard: bool = False
    getglobaladamicadar: bool = False
    getprefattachment: bool = False
    getshortestpathpbc: bool = False
    sp_box_size: float = 10.0
    getkatz: bool = False
    getEigenvectorCentrality: bool = False
    getBetweennessCentralityCorrelation: bool = False
    getMinCutValue: bool = False
    getMaximumFlow: bool = False
    getLaplacianEigenvectorSimilarity: bool = False
    getFiedlerVectorSimilarity: bool = False
    getGraphDistanceWeightedByBondOrder: bool = False
    getBetweennessCentralityOfPathways: bool = False
    getRingsInSharedPath: bool = False
    global_fingerprint_type: str = 'Morgan'
    global_similarity_metric: str = 'Tanimoto'

class SplitParams:
    split_type: str
    train_size: float
    test_size: float
    n_splits: int
    fold_shuffle: bool
    random_state: int
    smiles: str
    target: str
    astartes: Optional[dict] = None

@dataclass
class AstartesParams:
    hopts: dict
    fingerprint: str
    fingerprint_args: dict

@dataclass
class LoaderParams:
    data_path: str
    exclude: Optional[str] = None
    test_only: bool = False
    root: str = ''
    class_choice: Optional[str] = None
    randomize: bool = False
    fold: Optional[int] = None
    foldtype: Optional[str] = None
    size: Optional[int] = None
    target: Optional[str] = None
    additionals: Optional[List[str]] = None
    addons: bool = False
    molecular: bool = False
    batch_size: int = 32


@dataclass
class PredictParams:
    molecular: bool = False
    target: Union[str, List[str]] = "default_target"
    batch_size: int = 32
    model_type: str = "default_model"
    additionals: Optional[Union[str, List[str]]] = None
    Norm: Optional[str] = None
    hasaddons: bool = False
    Embed: bool = False
    AttnMaps: bool = False
    tweights: Optional[List[float]] = None
    save_path: str = "default_save_path.csv"



@dataclass
class Params:
    batch_size: int = 32
    additionals: Optional[Union[List[str], omegaconf.listconfig.ListConfig]] = None
    targets: Optional[Union[List[str], omegaconf.listconfig.ListConfig]] = None
    hasaddons: bool = False
    molecular: bool = False
    scaler: Type[BaseEstimator] = StandardScaler
    scaler_model: Optional[str] = None
    root: str = './'
    normtarget: bool = False
    model_type: str = 'Hr'
@dataclass
class BaseParams:
    weightsandbiases: bool = False
    wandbproject: str = ''
    wandbname: str = ''
    setup: str = 'cpu'
    parallel: bool = False
    gpu: int = 0
    save_path: str = ''
    startpoint: str = 'Retrain'
    base_model: Optional[str] = None
    ablation_EGAT_model: Optional[str] = None
    ablation_NN_model: Optional[str] = None
    loss: str = 'CrossEntropy'
    data_path: str = ''
    hasaddons: bool = False
    molecular: bool = False
    defaults: Optional[dict] = None
    learning_rate_min: float = 0.0
    momentum_orig: float = 0.9
    lr_decay: float = 0.1
    step_size: int = 10
    epochs: int = 100
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
    loss_agg: str = 'Arith'
    loss_weights: Optional[List[float]] = None
    save_style: str = 'best'


