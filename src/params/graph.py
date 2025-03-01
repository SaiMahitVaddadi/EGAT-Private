from dataclasses import dataclass
from typing import List, Optional, Union, Literal

@dataclass
class BaseGraphParams:
    getradical: Optional[str] = None
    stereo_full: Optional[bool] = False
    acidbase: Optional[str] = None


@dataclass
class BaseReactionGraphParams:
    mappingfunction: str
    rxnmapper: str
    totalmapping: Optional[bool] = False

@dataclass
class NeighborParams:
    neighbor: str = 'all'

@dataclass
class RingParams:
    removeringinfo: bool = False
    removearomaticity: bool = False

@dataclass
class ElectronParams:
    getradical: bool = False
    removehybridinfo: bool = False
    useFullHyb: bool = False

@dataclass
class ChargeParams:
    getelectronegativity: bool = False
    removeformalchargeinfo: bool = False
    charge: str = 'Gasteiger'

@dataclass
class AcidBaseParams:
    getacidbaseinfo: bool = False

@dataclass
class RDParams:
    getspiro: bool = False
    getbridgehead: bool = False
    getrotatablebonds: bool = False

@dataclass
class StereoParams:
    removechiralinfo: bool = False
    removeconjinfo: bool = False
    removestereoinfo: bool = False

@dataclass
class BondParams:
    removebondorderinfo: bool = False

@dataclass
class BondGeometryParams:
    getbondlength: bool = False
    getatomdistance: bool = False
    getbondangle: str = 'none'
    getdihedral: str = 'none'
    getbondmidpoint: bool = False

@dataclass
class BRICSParams:
    getbrics: bool = False
    checkbricsbond: bool = False

@dataclass
class AtomGeometryParams:
    removecoordinationinfo: bool = False
    getdistancetocenterofmass: bool = False
    getsterichindrance: bool = False
    getasa: bool = False
    getgaussiancurvature: bool = False
    getmolecularshapeindex: bool = False
    getdistancetoconvexhull: bool = False

@dataclass
class ReactiveAtomParams:
    removereactiveinfo: bool = False
    addneighboringreactives: bool = False

@dataclass
class ReactiveAtomChangeParams:
    gethybridizationchange: bool = False
    getdegreecentrality: bool = False
    getvalencychange: bool = False
    getoxidationreduction: bool = False
    getbondordersumchange: bool = False
    getneighborhoodchangeratio: bool = False
    getLocalAtomicEnvironmentSimilarity: bool = False
    global_fingerprint_type: str = 'Morgan'
    global_similarity_metric: str = 'Tanimoto'

@dataclass
class ReactiveBondParams:
    adddisttoreactingbonds: bool = False
    getbrics: bool = False

@dataclass
class NonBondedParams:
    addmissingbonds: str = 'hbonds'
    addcho: bool = False
    adddihydrogenbonds: bool = False
    addcationpi: bool = False
    addpipistack: bool = False
    addhalogenbonds: bool = False
    addmetallophilic: bool = False
    addelectrostatic: bool = False
    addeneg: bool = False

@dataclass
class GlobalBondParams:
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
    getLocalAtomicEnvironmentSimilarity: bool = False
    sp_box_size: float = 10.0

@dataclass
class HydrogenBondParams:
    check_hbond: bool = False

@dataclass
class GlobalReactionBondParams:
    getshortestpathchange: bool = False
    getcommonneighborcountchange: bool = False
    getrandomwalkchange: bool = False
    getbondpathorderchange: bool = False
    getsharedfunctionalgroupchange: bool = False
    getconnectivitypathdifference: bool = False
    getreactivitydistance: bool = False
    getelectronflowcorrelation: bool = False

class MoleculeFeaturizerParams:
    removeelementinfo: bool = False

@dataclass
class MoleculeGlobalParams:
    addcho: bool = False
    adddihydrogenbonds: bool = False
    addcationpi: bool = False
    addpipistack: bool = False
    addhalogenbonds: bool = False
    addmetallophilic: bool = False
    addelectrostatic: bool = False
    addeneg: bool = False
    addmissingbonds: Literal['none', 'global', 'hbonds', 'all'] = 'none'

@dataclass
class ReactionBaseParams:
    mappingfunction: str
    rxnmapper: Any
    totalmapping: Optional[bool] = False

class ReactionHelperParams:
    oldbondencode: bool = False
    addneighboringreactives: bool = False


class GraphParams(BaseGraphParams,BaseReactionGraphParams, NeighborParams, RingParams, ElectronParams,
                  ChargeParams, AcidBaseParams, RDParams, StereoParams, BondParams,
                  BondGeometryParams, BRICSParams, AtomGeometryParams, ReactiveAtomParams,
                  ReactiveAtomChangeParams, ReactiveBondParams, NonBondedParams,
                  GlobalBondParams, HydrogenBondParams, GlobalReactionBondParams,MoleculeFeaturizerParams,MoleculeGlobalParams,ReactionBaseParams,ReactionHelperParams):
    pass



