from dataclasses import dataclass
from typing import List, Optional, Union, Literal

from ..graph.molecular.Molecule import MoleculeFeaturizerParams
from ..graph.base.base import BaseReactionParams
from ..graph.base.reaction import ReactionBaseParams
from ..graph.base.information.acidbase import AcidBaseParams
from ..graph.base.information.atomgeom import AtomGeometryParams
from ..graph.base.information.bond import BondParams
from ..graph.base.information.brics import BRICSParams
from ..graph.base.information.dijkstra import DijkstraParams
from ..graph.base.information.electron import ElectronParams
from ..graph.base.information.fused import FusedInformationParams
from ..graph.base.information.globalatom import GlobalAtomParams
from ..graph.base.information.globalbond import GlobalBondParams
from ..graph.base.information.globalreaction import ReactionParams
from ..graph.base.information.hbond import HydrogenBondParams
from ..graph.base.information.interaction import InteractionParams
from ..graph.base.information.kallisto import KallistoParams
from ..graph.base.information.locant import LocantParams
from ..graph.base.information.mordreddescs import MordredParams
from ..graph.base.information.nbi import NonBondedParams
from ..graph.base.information.neighbors import NeighborParams
from ..graph.base.information.randomwalk import RWParams
from ..graph.base.information.reactionatom import ReactiveAtomParams
from ..graph.base.information.reactionbond import ReactiveBondParams
from ..graph.base.information.ring import RingParams
from ..graph.base.information.stereo import StereoParams
from ..graph.base.information.sterimol import SterimolParams
from ..graph.molecular.MoleculeGlobal import MoleculeGlobalParams
from ..utils.descriptors.geometry import ConformerGeneratorParams
@dataclass
class GraphGenerationParams(MoleculeFeaturizerParams,BaseReactionParams,ReactionBaseParams,AtomGeometryParams,BondParams,AcidBaseParams,BRICSParams,
                            DijkstraParams,ElectronParams,FusedInformationParams,GlobalAtomParams,GlobalBondParams,HydrogenBondParams,InteractionParams,
                            KallistoParams,LocantParams,MordredParams,NonBondedParams,NeighborParams,ReactionParams,ReactiveAtomParams,ReactiveBondParams,
                            RWParams,RingParams,StereoParams,SterimolParams,MoleculeGlobalParams):
    conformer: ConformerGeneratorParams = ConformerGeneratorParams()
    