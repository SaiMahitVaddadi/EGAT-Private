
from dataclasses import dataclass,field
from typing import Literal,Optional,List,Union

@dataclass
class FeaturizerParams:
    getradical: Optional[Literal['RDKit', 'YARP']] = 'RDKit'
    neighbor: Optional[Literal['onlyH', 'onlyCHNO', 'onlyorganic', 'all']] = 'onlyCHNO'
    removeelementinfo: Optional[bool] = False
    removereactiveinfo: Optional[bool] = False
    removeringinfo: Optional[bool] = False
    removeformalchargeinfo: Optional[bool] = False
    removearomaticity: Optional[bool] = False
    removehybridinfo: Optional[bool] = False
    removechiralinfo: Optional[bool] = False
    getspiro: Optional[bool] = False
    getbridgehead: Optional[bool] = False
    getelectronegativity: Optional[bool] = False
    charge: Optional[Literal['Gasteiger', 'psi4', 'pyscf', 'gaussian', 'orca', 'ase', 'gemnet', 'chemprop', 'unimol', None]] = None
    getrotatablebonds: Optional[bool] = False
    removebondorderinfo: Optional[bool] = False
    removeconjinfo: Optional[bool] = False
    removestereoinfo: Optional[bool] = False
    addmissingbonds: Optional[str] = None
    addcho: Optional[bool] = False
    adddihydrogenbonds: Optional[bool] = False
    addcationpi: Optional[bool] = False
    addpipistack: Optional[bool] = False
    addhalogenbonds: Optional[bool] = False
    addmetallophilic: Optional[bool] = False
    addelectrostatic: Optional[bool] = False
    addeneg: Optional[bool] = False
    conformer: Optional[dict] = None
    useFullHyb: Optional[bool] = False
    rxnmapper: Optional[object] = None
    mappingfunction: Optional[str] = 'RXNMapper'
    totalmapping: Optional[bool] = False
    adddisttoreactingbonds: Optional[bool] = False
    addneighboringreactives: Optional[bool] = False
    oldbondencode: Optional[bool] = False
    addbonddistancechange: Optional[bool] = False
    getacidbaseinfo: Optional[bool] = False
    acidbase: Optional[str] = 'Lewis'

@dataclass
class DataLoaderParams:
    root: Optional[str] = './data'
    data_path: Optional[str] = './data'
    exclude: Optional[str] = None
    test_only: Optional[bool] = False
    class_choice: Optional[str] = None
    randomize: Optional[bool] = True
    fold: Optional[int] = 0
    foldtype: Optional[str] = 'random'
    size: Optional[int] = None
    target: Optional[str] = None
    additionals: Optional[List[str]] = field(default_factory=list)
    addons: Optional[bool] = False
    molecular: Optional[bool] = False
    batch_size: Optional[int] = 32


@dataclass
class AstartesParams:
    fingerprint: Optional[str] = None

@dataclass
class Params:
    split_type: Optional[str] = 'random'
    train_size: Optional[float] = .8
    random_state: Optional[int] = 42
    targets: Optional[Union[str, List[str]]] = None
    smiles: Optional[str] = None
    astartes: Optional[AstartesParams] = None