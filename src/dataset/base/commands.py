import os 
import pandas as pd 
import torch
import traceback
import dgl 
from torch.utils.data import Dataset
from rdkit import Chem
import omegaconf
from mordred import Calculator, descriptors

from ...splitting.split import Splitter
from ...utils.descriptors.egat.encodings import Encodings
from ...utils.database.csvfunctions import DenoteInputData
from ...utils.descriptors.fingerprint.threedimensional import GeometricFingerprint
from ...utils.descriptors.fingerprint.twodimensional import Fingerprint
from ...utils.descriptors.fingerprint.reaction import ReactionFingerprint
from ...utils.descriptors.fingerprint.rdkit import RDKitDescriptors
from ...utils.descriptors.fingerprint.descriptors import Descriptors
from ...graph.molecular.Molecule import MoleculeFeaturizer
from ...graph.molecular.MoleculeGeometry import MoleculeFeaturizerwithGeometry
from ...graph.reaction.ReactionGeometry import ReactionFeaturizerwithGeometry
from ...graph.reaction.ReactionGlobalGeometry import ReactionFeaturizerwithPaddingandGeometry
from ...graph.molecular.MoleculeGlobal import MoleculeFeaturizerwithPadding
from ...graph.reaction.ReactionGlobal import ReactionFeaturizerwithPadding
from ...graph.reaction.Reaction import ReactionFeaturizer
from ...tools.jepa.tools import mask_uv_vectors,mask_node_features,mask_node_features_v2,mask_edge_features,mask_edge_features_v2,mask_node_and_edges,mask_node_and_edges_v2,mask_node_and_edges_v3,mask_node_and_neighbors
from ...processing.normalize import DataNormalizer
from torch_geometric.data import Data
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict
from .basecommands import BaseDatasetCommands
from .dataset_setups import DatasetSetups
from ...utils.misc.RDKHelpers import getInchifromSMILES,RemoveMapping

@dataclass
class DatasetParams:
    root: str
    data_path: str
    rootfile: Union[str, List[str], Dict]
    exclude: Optional[List[int]] = None
    smiles: Union[str, List[str]] = "smiles"
    target: Union[str, List[str]] = "target"
    additional: Optional[Union[str, List[str]]] = None
    addons: Optional[Union[str, List[str]]] = None
    graph: str = "molecular"  # Options: 'molecular', 'reaction'
    dimension: str = "2d"  # Options: '2d', '3d'
    globalmode: bool = False
    randomize: bool = False
    size: Optional[int] = None
    split: Optional[int] = None
    n_splits: Optional[int] = None
    splittotrain: Optional[float] = None
    fold: Optional[int] = None
    denotation: Optional[object] = None
    normalize_step: str = "all"  # Options: 'all', 'subset'
    target_normalizer: Optional[str] = None
    target_normalizer_params: Optional[Dict] = field(default_factory=dict)
    additional_normalizer: Optional[str] = None
    additional_normalizer_params: Optional[Dict] = field(default_factory=dict)
    mode: str = "dgl"  # Options: 'dgl', 'pyg'
    getradical: str = "N"  # Options: 'YARP', 'N'
    conformer: object = field(default_factory=lambda: type("Conformer", (), {"nconfs": 1}))
    jepa: bool = False
    jepa_masking: Optional[str] = None  # Options: 'node', 'edge', 'node+edge', etc.
    jepa_num_nodes: Optional[int] = None
    jepa_num_edges: Optional[int] = None
    jepa_neighbors: Optional[int] = None
    multicomponent: bool = False
    cache_size: int = 100

class DatasetCommands(DatasetSetups,BaseDatasetCommands):
    def __init__(self, arguments,split=None):
        self.params = arguments
        self.split = split
        self.SetupInits()
        self.checkmolecular()
        self.InitialSetup() 

    def __useinfolist(self):
        if self.params.getradical == 'YARP':
            if self.featurizer.bond_mats > 1:
                return True
            elif self.params.conformer.nconfs > 1:
                return True
            else:
                return False
        else:
            if self.params.conformer.nconfs > 1:
                return True
            else:
                return False

    
    def GetRow(self,Rind):
        if isinstance(Rind,str):
            rxn = self.data.loc[Rind,:]
        else:
            rxn = self.data.iloc[Rind,:]
        return rxn

    def NormTargets(self,rxn):
        return self._setnormcolumnsinrxn(rxn,self.target_normalizer,self.params.target)

    def NormAdditionals(self,rxn):
        return self._setnormcolumnsinrxn(rxn,self.additional_normalizer,self.params.additional)
    
    def AddTargets(self,rxn):
        self._setnormstoinfo(rxn,self.params.target,self.params.target_normalizer)
    
    def AddAdditionals(self,rxn):
        if self.params.additional is not None:
            self._setnormstoinfo(rxn,self.params.additional,self.params.additional_normalizer)
    
    def AddINCHI(self,rxn):
        #get the functions not there. 
        if 'reaction' in self.params.graph:
            if isinstance(self.params.smiles,str):
                self.info = self._reactioninchiandsmilessetup(rxn,self.info,self.params.smiles)
            else:
                self.info = self._multireactioninchiandsmilessetup(rxn,self.info,self.params.smiles)
            
        elif 'molecular' in self.params.graph:
            if isinstance(self.params.smiles,str):
                Rsmiles = rxn[self.params.smiles]
                self.info = self._createinchisandsmiles(Rsmiles,self.info,'R')
            elif isinstance(self.params.smiles,list) or isinstance(self.params.smiles,omegaconf.listconfig.ListConfig):
                for smi in self.params.smiles:
                    Rsmiles = rxn[smi]
                    self.info = self._createinchisandsmiles(Rsmiles,self.info,'R',smi)
        
    def ChooseFeaturizerStringCase(self,rxn):
        if 'molecular' in self.params.graph:
            self.featurizer = self._featurization(rxn[self.params.smiles])
        if 'reaction' in self.params.graph:
            self.featurizer = self._featurization(rxn[self.params.smiles],reaction=True)
        self.featurizer.run()

    def ChooseFeaturizerListCase(self,rxn):
        self.featurizer = dict()
        for i in range(len(self.params.smiles)):
            if 'molecular' in self.params.graph:
                self.featurizer[self.params.smiles[i]] = self._featurization(rxn[self.params.smiles[i]])
            if 'reaction' in self.params.graph:
                self.featurizer[self.params.smiles[i]] = self._featurization(rxn[self.params.smiles[i]],reaction=True)
            self.featurizer[self.params.smiles[i]].run()
    
    def ChooseFeaturizer(self,rxn):
        if isinstance(self.params.smiles,str):
            self.ChooseFeaturizerStringCase(rxn)
        elif isinstance(self.params.smiles,list) or isinstance(self.params.smiles,omegaconf.listconfig.ListConfig):
            self.ChooseFeaturizerListCase(rxn)
  
    def AddFeatures(self):
        self._createinfolist()
        #Change this for the cases with multiple resonances with the YARP case and nconfs >1
        if self.params.getradical == 'YARP':
            if self.featurizer.bond_mats == 1:
                self._singlegeomyarpcase()
            else:
                self._multigeomyarpcase()
        else:
            if self.params.conformer.nconfs > 1:
                self._multigeomyarpcase()
            else:
                self._singlegeomyarpcase()

    def InfoorInfoList(self,Rind):
        self.info['Indices'] = Rind
        rxn = self.GetRow(Rind)
        self.AddTargets(rxn)
        self.AddAdditionals(rxn)
        self.AddINCHI(rxn)
        self.ChooseFeaturizer(rxn)
        self.AddFeatures()

    def Convert(self,Rind):        
        self.info = {}
        try:
            self.InfoorInfoList(Rind)
        except:
            print(self.root + '--'+ str(Rind) + ' failed')
            print(traceback.print_exc())
            return None
                 
    def CreateGraph(self, index):
        if self.params.mode == 'dgl':
            gR,gP = self._creategraphsdgl()
        if 'reaction' in self.params.graph:
            return gR,gP
        elif 'molecular' in self.params.graph:
            return gR
        elif self.params.mode == 'pyg':
            raise NotImplementedError('Pytorch Geometric is not implemented yet')

    def AddGraphsToSample(self,samples,infolistusage=False,index=0):
        if 'reaction' in self.params.graph:
            gR,gP = self.CreateGraph(index)
            samples += [gR,gP]
        elif 'molecular' in self.params.graph:
            gR = self.CreateGraph(index)
            samples += [gR]
        samples += [[self._smilesandinchisampler(infolistusage)]]
        return samples

    def Sample(self,index):
        infolistusage = self.__useinfolist()
        print(infolistusage)
        self.InfoorInfoList(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        self.samples = self.AddGraphsToSample(self.samples)
        self.samples += self._sampletargets(infolistusage)
        if self.params.additional is not None: self.samples += self._sampleadditionals(infolistusage)
        if self.params.addons is not None: self.samples += self.Addons()
        return self.samples
    
    def FingerprintModelSampler(self,index):
        infolistusage = self.__useinfolist()
        self.InfoorInfoList(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        self.samples += self._sampletargets(infolistusage)
        if self.params.additional is not None: self.samples += self._sampleadditionals(infolistusage)
        if self.params.addons is not None:self.samples += self.Addons()
        return self.samples

    def Addons(self):
        # Add what the Add-Ons are
        if self.params.addons is not None: 
            self.obtainaddons()
            samples = self.tensorizeinfo()
            return samples 
                # save in cache

    def RunJEPA(self,graph):
        if self.params.jepa:
            if self.params.jepa_masking == 'node':
                mask_node_features(graph)
            elif self.params.jepa_masking == 'edge':
                mask_edge_features(graph)
            elif self.params.jepa_masking == 'node+edge':
                mask_node_and_edges(graph)
            elif self.params.jepa_masking == 'node+edge_v2':
                mask_node_and_edges_v2(graph,self.params.jepa_num_nodes)
            elif self.params.jepa_masking == 'node+edge_v3':
                mask_node_and_edges_v3(graph,self.params.jepa_num_nodes,self.params.jepa_num_edges)
            elif self.params.jepa_masking == 'node+neighbors':
                mask_node_and_neighbors(graph,self.params.jepa_neighbors)
            elif self.params.jepa_masking == 'node_v2':
                mask_node_features_v2(graph, self.params.jepa_num_nodes)
            elif self.params.jepa_masking == 'edge_v2':
                mask_edge_features_v2(graph, self.params.jepa_num_edges)
    
    def __ItemException(self,index):
        print(self.root + '--'+ str(index) + ' failed')
        print(traceback.print_exc())
        return None

        