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
    neighbor: str = 'onlyCHNO'

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
    mappingfunction: str = 'RXNMapper'
    totalmapping: Optional[bool] = True

class ReactionHelperParams:
    oldbondencode: bool = False
    addneighboringreactives: bool = False


@dataclass
class ConformerGeneratorParams:
    nconfs: int = 1
    method: Optional[str] = None
    seed: int = 1
    verbose: bool = False
    theory: str = 'UFF'
    tmp: str = 'tmp'
    window: Optional[int] = None
    engine: str = 'ANI2x'
    gpu: bool = False
    enumerate_tautomer: bool = True
    tauto_engine: str = "rdkit"
    pKaNorm: bool = True
    enumerate_isomer: bool = True
    max_confs: Optional[int] = None
    patience: int = 1000
    opt_steps: int = 5000
    convergence_threshold: float = .003
    threshold: float = .3
    genmode: str = 'confgen'
    geomopt: bool = False
    thermo: bool = False
    tauto_k: Optional[int] = None
    tauto_window: Optional[int] = None
    opt_tol: float = .0002


    infile: str = 'input.xyz'
    charge: int = 0
    uhf: int = 0
    solvation: Optional[str] = None
    optlev: Optional[str] = None
    sampling: Optional[str] = None
    sites: Optional[str] = None
    mdlen: Optional[str] = None
    shake: Optional[str] = None
    tstep: Optional[str] = None
    mddump: Optional[str] = None
    vbdump: Optional[str] = None
    zsort: bool = False
    genzsort: bool = True
    norotmd: bool = False
    tnmd: Optional[str] = None
    mrest: Optional[str] = None
    hflip: bool = True
    maxflip: Optional[str] = None
    gcspeed: Optional[str] = None
    props: Optional[str] = None
    origin: bool = True
    keepdir: bool = False
    noreftopo: bool = False
    noopt: bool = False
    wall: Optional[str] = None
    scthr: Optional[str] = None
    ssthr: Optional[str] = None
    trange: Optional[str] = None
    ptot: Optional[str] = None
    fscal: Optional[str] = None
    sthr: Optional[str] = None
    ithr: Optional[str] = None
    cinp: Optional[str] = None
    cbonds: Optional[str] = None
    cheavy: Optional[str] = None
    clight: Optional[str] = None
    fc: Optional[str] = None
    mdopt: Optional[str] = None
    screen: Optional[str] = None
    rrhoav: Optional[str] = None
    thermo: Optional[str] = None
    nanoreactor: Optional[str] = None
    solvent: Optional[str] = None
    testtopo: Optional[str] = None
    inputfile: Optional[str] = None

    tmp: str = 'tmp'
    solvent: str = 'h2o'
    run: str = 'grow'
    nsolv: Optional[int] = None
    nopreopt: bool = False
    keepdir: bool = False
    gfn1: bool = False
    gfn2: bool = False
    gfnff: bool = False
    samerand: bool = False
    chrg: Optional[int] = None
    uhf: Optional[int] = None
    wscal: Optional[float] = None
    fixsolute: bool = False
    nofix: bool = False
    xtbiff: bool = False
    normdock: bool = False
    directed: Optional[str] = None
    qcgmtd: bool = False
    ncimtd: bool = False
    mtd: bool = False
    md: bool = False
    enslvl: Optional[str] = None
    mdlen: Optional[str] = None
    mddump: Optional[str] = None
    tstep: Optional[str] = None
    vbdump: Optional[str] = None
    norotmd: bool = False
    tnmd: Optional[str] = None
    mreset: Optional[str] = None
    fin_opt_gfn2: bool = False
    nocff: bool = False
    esolv: bool = False
    nclus: Optional[int] = None
    freqlvl: Optional[str] = None
    freqscal: Optional[float] = None

    ewin: Optional[float] = None
    rthr: Optional[float] = None
    ethr: Optional[float] = None
    bthr: Optional[float] = None
    pthr: Optional[float] = None
    nmr: bool = False
    eqv: bool = False
    athr: Optional[float] = None
    temp: Optional[float] = None
    esort: bool = False
    nowr: bool = False
    subrmsd: bool = False
    notopo: Optional[str] = None
    cluster: Optional[str] = None
    pccap: Optional[float] = None
    nopcmin: bool = False
    pcaex: Optional[float] = None


class GraphParams(BaseGraphParams,BaseReactionGraphParams, NeighborParams, RingParams, ElectronParams,
                  ChargeParams, AcidBaseParams, RDParams, StereoParams, BondParams,
                  BondGeometryParams, BRICSParams, AtomGeometryParams, ReactiveAtomParams,
                  ReactiveAtomChangeParams, ReactiveBondParams, NonBondedParams,
                  GlobalBondParams, HydrogenBondParams, GlobalReactionBondParams,MoleculeFeaturizerParams,MoleculeGlobalParams,ReactionBaseParams,ReactionHelperParams):
    
    def __init__(self):
        self.conformer = ConformerGeneratorParams()




