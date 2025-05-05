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


class BaseDatasetCommands:
    def __init__(self, arguments):
        self.params = arguments
    
    def _setnormcolumnsinrxn(self,rxn,normalizer,column):
        if normalizer is not None: res = normalizer.transform(rxn[column])
        if isinstance(column,list) or isinstance(column,omegaconf.listconfig.ListConfig):
            for i,outputs in enumerate(column):
                rxn[f'{outputs}_normed'] = res[i]
        else:
            rxn[f'{column}_normed'] = res
        return rxn

    def _setnormstoinfo(self,rxn,column,normalizer):
        if isinstance(column,list) or isinstance(column,omegaconf.listconfig.ListConfig):
            for outputs in column:
                if normalizer is not None:
                    self.info[outputs] = rxn[f'{outputs}_normed']
                else:
                    self.info[outputs] = rxn[outputs]
        else:
            if normalizer is not None:
                self.info[column] = rxn[f'{column}_normed']
            else:
                self.info[column] = rxn[column]
    
    def _createinchisandsmiles(self,smiles,info,prequel='R',sequel=None):
        if sequel is not None:
            inchikeyname = f'{prequel}inchi' + '_' + sequel
            smileskeyname = f'{prequel}smiles' + '_' + sequel
        else:
            inchikeyname = f'{prequel}inchi'
            smileskeyname = f'{prequel}smiles'
        info[inchikeyname] = getInchifromSMILES(smiles)
        Nsmiles = RemoveMapping(smiles)
        info[smileskeyname] = Chem.MolToSmiles(Nsmiles)
        return info
    
    def _reactioninchiandsmilessetup(self,rxn,info,smiles):
        RPsmiles = rxn[smiles].split('>>')
        Rsmiles = RPsmiles[0]
        Psmiles = RPsmiles[1]        
        self._createinchisandsmiles(Rsmiles,info,'R')
        self._createinchisandsmiles(Psmiles,info,'P')
        return info
    
    def _multireactioninchiandsmilessetup(self,rxn,info,smiles):
        for smi in smiles:
            RPsmiles = rxn[smiles].split('>>')
            Rsmiles = RPsmiles[0]
            Psmiles = RPsmiles[1]        
            self._createinchisandsmiles(Rsmiles,info,'R',smi)
            self._createinchisandsmiles(Psmiles,info,'P',smi)
        return info
    
    def _featurization(self,smiles,reaction=False):
        if reaction: prestring = 'Reaction'
        else: prestring = 'Molecule'
        if self.params.dimension == '2d':
            if self.params.globalmode:
                featurizer = getattr(prestring + 'FeaturizerwithPadding')(smiles,self.params)
            else:
                featurizer = getattr(prestring + 'Featurizer')(smiles,self.params)
        elif self.params.dimension == '3d':
            if self.params.globalmode:
                featurizer = getattr(prestring + 'FeaturizerwithPadding')(smiles,self.params)
            else:
                featurizer = getattr(prestring + 'FeaturizerwithGeometry')(smiles,self.params)
        return featurizer
    
    def _createinfolist(self):
        if self.params.getradical == 'YARP':
            if self.featurizer.bond_mats > 1:
                self.infolist = []
        else:
            if self.params.conformer.nconfs > 1:
                self.infolist = []

    def __edgedictnames(self,sequel=None):
        uname = 'u'
        vname = 'v'
        if sequel is not None:
            uname = f'u_{sequel}'
            vname = f'v_{sequel}'
        return uname,vname
    
    def __atomfromdictnames(self,reaction=False,sequel=None):
        atomFname = 'atom_F'
        if reaction:
            if sequel is not None:
                atomFRname = f'{atomFname}_R_{sequel}'
                atomFPname = f'{atomFname}_P_{sequel}'
            else:
                atomFRname = f'{atomFname}_R'
                atomFPname = f'{atomFname}_P'
        else:
            if sequel is not None:
                atomFRname = f'{atomFname}_R_{sequel}'
                atomFPname = f'{atomFname}_R_{sequel}'
            else:
                atomFRname = f'{atomFname}_R'
                atomFPname = f'{atomFname}_R'
        return atomFRname,atomFPname
    
    def __bondfromdictnames(self,reaction=False,sequel=None):
        atomFname = 'bond_F'
        if reaction:
            if sequel is not None:
                atomFRname = f'{atomFname}_R_{sequel}'
                atomFPname = f'{atomFname}_P_{sequel}'
            else:
                atomFRname = f'{atomFname}_R'
                atomFPname = f'{atomFname}_P'
        else:
            if sequel is not None:
                atomFRname = f'{atomFname}_R_{sequel}'
                atomFPname = f'{atomFname}_R_{sequel}'
            else:
                atomFRname = f'{atomFname}_R'
                atomFPname = f'{atomFname}_R'
        return atomFRname,atomFPname
    
    def __geomfromdictnames(self,reaction=False,sequel=None):
        atomFname = 'Geom'
        if reaction:
            if sequel is not None:
                atomFRname = f'{atomFname}_R_{sequel}'
                atomFPname = f'{atomFname}_P_{sequel}'
            else:
                atomFRname = f'{atomFname}_R'
                atomFPname = f'{atomFname}_P'
        else:
            if sequel is not None:
                atomFRname = f'{atomFname}_R_{sequel}'
                atomFPname = f'{atomFname}_R_{sequel}'
            else:
                atomFRname = f'{atomFname}_R'
                atomFPname = f'{atomFname}_R'
        return atomFRname,atomFPname

    def __reactioninfo(self,info,featurizer,uname,vname,atomFRname,atomFPname,bondFRname,bondFPname):
        info[uname] = featurizer.reactant.edges_u
        info[vname] = featurizer.reactant.edges_v
        info[atomFRname] = featurizer.reactant.atom_features
        info[bondFRname] = featurizer.reactant.bond_features
        info[atomFPname] = featurizer.product.atom_features
        info[bondFPname] = featurizer.product.bond_features
        return info
    
    def __molecularinfo(self,info,featurizer,uname,vname,atomFRname,bondFRname):
        info[uname] = featurizer.edges_u
        info[vname] = featurizer.edges_v
        info[atomFRname] = featurizer.atom_features
        info[bondFRname] = featurizer.bond_features
        return info

    def __geominfo(self,info,featurizer,atomFRname,atomFPname,reaction=False):
        geomusecase,geomfeat = self.__usegeometryvector()
        if reaction:
            if geomusecase:
                info[atomFRname] = getattr(featurizer.reactant,geomfeat)
                info[atomFPname] = getattr(featurizer.product,geomfeat)
        else:
            if geomusecase:
                info[atomFRname] = getattr(featurizer,geomfeat)
        return info
    
    def _addfeatstoinfo(self,info,featurizer,reaction=False,sequel=None):
        uname,vname = self.__edgedictnames(sequel)
        atomFRname,atomFPname = self.__atomfromdictnames(reaction,sequel)
        bondFRname,bondFPname = self.__bondfromdictnames(reaction,sequel)
        atomgeomnameR,atomgeomnameP = self.__geomfromdictnames(reaction,sequel) 
        if 'reaction' in self.params.graph:
            info = self.__reactioninfo(info,featurizer,uname,vname,atomFRname,atomFPname,bondFRname,bondFPname)
        elif 'molecular' in self.params.graph:
            info = self.__molecularinfo(info,featurizer,uname,vname,atomFRname,bondFRname)
        info = self.__geominfo(info,featurizer,atomgeomnameR,atomgeomnameP,reaction)
        return info 

    def _singlegeomyarpcase(self):
        if 'reaction' in self.params.graph:
            if isinstance(self.params.smiles,str):
                self.info = self._addfeatstoinfo(self.info,self.featurizer,reaction=True)
            elif isinstance(self.params.smiles,list) or isinstance(self.params.smiles,omegaconf.listconfig.ListConfig):
                for i,smi in enumerate(self.params.smiles):
                    self.info = self._addfeatstoinfo(self.info,self.featurizer[i],reaction=True,sequel=smi)
        elif 'molecular' in self.params.graph:
            if isinstance(self.params.smiles,str):
                self.info = self._addfeatstoinfo(self.info,self.featurizer)
            elif isinstance(self.params.smiles,list) or isinstance(self.params.smiles,omegaconf.listconfig.ListConfig):
                for i,smi in enumerate(self.params.smiles):
                    self.info = self._addfeatstoinfo(self.info,self.featurizer[i],sequel=smi)
    
    def __usemultigeom(self):
        if self.params.getradical == 'YARP':
            if self.params.dimension == '2d':   
                return True,'atom_features_dict','bond_features_dict'
            elif self.params.dimension == '3d':
                return True,'atom_features_confs','bond_features_confs'
        
        if self.params.conformer.nconfs > 1 and self.params.dimension == '3d':
            return True,'atom_features_confs','bond_features_confs'
        
        return False,'',''

    def __usegeometryvector(self):
        if self.params.getradical == 'YARP':
            if self.params.dimension == '2d':   
                return False,''
            elif self.params.dimension == '3d':
                return True,'atom_geometry_feature_confs_list'
        
        if self.params.conformer.nconfs > 1 and self.params.dimension == '3d':
            return True,'atom_geometry_feature_confs_list','bond_features_confs'
        elif self.params.conformer.nconfs == 1 and self.params.dimension == '3d':
            return True,'atom_geometry_features'
        
        return False,''

    def __getallnames(self,reaction=False):
        unames = []
        vnames = []
        atomFRnames = []
        atomFPnames = []
        bondFRnames = []
        bondFPnames = []
        Ratomgeomnames = []
        Patomgeomnames = []
        if isinstance(self.params.smiles,str):
            unames,vnames = self.__edgedictnames()
            atomFRnames,atomFPnames = self.__atomfromdictnames(reaction)
            bondFRnames,bondFPnames = self.__bondfromdictnames(reaction)
            Ratomgeomnames,Patomgeomnames = self.__geomfromdictnames(reaction)
        else:
            for smi in self.params.smiles:
                uname,vname = self.__edgedictnames(smi)
                atomFRname,atomFPname = self.__atomfromdictnames(reaction,smi)
                bondFRname,bondFPname = self.__bondfromdictnames(reaction,smi)
                atomgeomnameR,atomgeomnameP = self.__geomfromdictnames(reaction,smi) 
                unames.append(uname)
                vnames.append(vname)
                atomFRnames.append(atomFRname)
                atomFPnames.append(atomFPname)
                bondFRnames.append(bondFRname)
                bondFPnames.append(bondFPname)
                Ratomgeomnames.append(atomgeomnameR)
                Patomgeomnames.append(atomgeomnameP)
        return unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,Ratomgeomnames,Patomgeomnames
    
    def __setupinfo(self,adict,keys,featurizers,attribute,ind=None,reaction=None):
        if ind is None:
            if isinstance(keys,list):
                for i,key in enumerate(keys):
                    if not reaction: adict[key] = getattr(featurizers[i],reaction,attribute)
                    else: adict[key] = getattr(featurizers[i],attribute)
            else:
                if not reaction: adict[keys] = getattr(featurizers[i],reaction,attribute)
                else: adict[keys] = getattr(featurizers[i],attribute)
        else:
            if isinstance(keys,list):
                for i,key in enumerate(keys):
                    if not reaction: adict[key] = getattr(featurizers[i],reaction,attribute)[ind]
                    else: adict[key] = getattr(featurizers[i],attribute)[ind]
            else:
                if not reaction: adict[keys] = getattr(featurizers[i],reaction,attribute)[ind]
                else: adict[keys] = getattr(featurizers[i],attribute)[ind]
        return adict
    
    def __getconfs(self,atomfeat,featurizer,reaction=None,len=1):
        if len == 1:
            if reaction is not None: 
                confs = getattr(featurizer.reactant,atomfeat)
            else:
                confs = getattr(featurizer,atomfeat)
        else:
            if reaction is not None: 
                confs = getattr(featurizer[self.params.smiles[0]].reactant,atomfeat)
            else:
                confs = getattr(featurizer[self.params.smiles[0]],atomfeat)
        return confs

    def __reactioninfomultigeom(self,info,featurizer,unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames):
        usecase,atomfeat,bondfeat = self.__usemultigeom()
        geomusecase,geomfeat = self.__usegeometryvector()
        confs = self.__getconfs(atomfeat,featurizer,reaction=True,len=len(self.params.smiles) if isinstance(self.params.smiles,list) else 1)
        infolist = [] 
        
        if usecase:
            _info = info.copy()
            _info = self.__setupinfo(_info,unames,featurizer,'edges_u','reactant')
            _info = self.__setupinfo(_info,vnames,featurizer,'edges_v','reactant')
            for i in range(len(confs)):
                _info = self.__setupinfo(_info,atomFRnames,featurizer,atomfeat,i,'reactant')
                _info = self.__setupinfo(_info,bondFRnames,featurizer,bondfeat,i,'reactant')
                if geomusecase: 
                    _info = self.__setupinfo(_info,'R_Geom',featurizer,geomfeat,i,'reactant')
                for j in range(len(confs)):
                    _info = self.__setupinfo(_info,atomFPnames,featurizer,atomfeat,j,'product')
                    _info = self.__setupinfo(_info,bondFPnames,featurizer,bondfeat,j,'product')
                    if geomusecase:
                        _info = self.__setupinfo(_info,'P_Geom',featurizer,geomfeat,j,'product')
                    infolist.append(_info.copy())        
        return infolist

    def __moleculeinfomultigeom(self,info,featurizer,unames,vnames,atomFRnames,bondFRnames):
        usecase,atomfeat,bondfeat = self.__usemultigeom()
        geomusecase,geomfeat = self.__usegeometryvector()
        confs = self.__getconfs(atomfeat,featurizer,reaction=False,len=len(self.params.smiles) if isinstance(self.params.smiles,list) else 1)
        infolist = [] 
        confs = getattr(featurizer,atomfeat)
        if usecase:
            _info = info.copy()
            _info = self.__setupinfo(_info,unames,featurizer,'edges_u')
            _info = self.__setupinfo(_info,vnames,featurizer,'edges_v')
            for i in range(len(confs)):
                _info = self.__setupinfo(_info,atomFRnames,featurizer,atomfeat,i)
                _info = self.__setupinfo(_info,bondFRnames,featurizer,bondfeat,i)
                if geomusecase: 
                    _info = self.__setupinfo(_info,'R_Geom',featurizer,geomfeat,i)
                infolist.append(_info.copy())
        return infolist

    def _addfeatstoinfolist(self,info,infolist,featurizer,reaction=True):
        unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames = self.__getallnames(reaction)
        if 'reaction' in self.params.graph:
            infolist += self.__reactioninfomultigeom(info,featurizer,unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames)
        elif 'molecular' in self.params.graph:
            infolist += self.__moleculeinfomultigeom(info,featurizer,unames,vnames,atomFRnames,bondFRnames)
        return infolist 
    
    def _multigeomyarpcase(self):
        self.infolist = self._addfeatstoinfolist(self.info,self.infolist,self.featurizer,reaction=True)
    
    def __grabuv(self,info,uname,vname):
        u, v = torch.Tensor(info[uname]).int(),torch.Tensor(info[vname]).int()
        if self.params.jepa: u,v = mask_uv_vectors(u,v)
        return u,v

    def _dglgraphcreation(self,u,v):
        gR   = dgl.graph((u,v))
        gP   = dgl.graph((u,v))
        return gR,gP    

    def _dglreactiongraphfcn(self,info,gR,gP,atomFRname,atomFPname,bondFRname,bondFPname,atomgeomnameR,atomgeomnameP):
        gR.ndata['x'] = torch.Tensor(info[atomFRname])
        gR.edata['x'] = torch.Tensor(info[bondFRname])
        gP.ndata['x'] = torch.Tensor(info[atomFPname])
        gP.edata['x'] = torch.Tensor(info[bondFPname])
        gR = self._addgeomstodglgraph(gR,atomgeomnameR,info)
        gP = self._addgeomstodglgraph(gP,atomgeomnameP,info)
        if self.params.jepa: chosen_index = self.RunJEPA(gR)
        if self.params.jepa: chosen_index = self.RunJEPA(gP)
        return gR,gP
    
    def _dglmoleculargraphfcn(self,info,gR,atomFRname,bondFRname,atomgeomnameR):
        gR.ndata['x'] = torch.Tensor(info[atomFRname])
        gR.edata['x'] = torch.Tensor(info[bondFRname])
        gR = self._addgeomstodglgraph(gR,atomgeomnameR,info)
        if self.params.jepa: chosen_index = self.RunJEPA(gR)
        return gR
    
    def _addgeomstodglgraph(self,g,atomgeomprop,info):
        if self.params.dimension == '3d':
            g.ndata['g'] = torch.Tensor(info[atomgeomprop])
        return g


    def _multicompgraphcreation(self,info,unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP):
        gRs = []
        gPs = []
        for i in range(len(unames)):
            u,v = self.__grabuv(info,unames[i],vnames[i])
            gR,gP = self._dglgraphcreation(u,v)
            if 'reaction' in self.params.graph:
                gR,gP = self._dglreactiongraphfcn(info,gR,gP,atomFRnames[i],atomFPnames[i],bondFRnames[i],bondFPnames[i],atomgeomnamesR[i],atomgeomnamesP[i])
                gRs.append(gR)
                gPs.append(gP)
            elif 'molecular' in self.params.graph:
                gR = self._dglmoleculargraphfcn(info,gR,atomFRnames[i],bondFRnames[i],atomgeomnamesR[i])
                gRs.append(gR)
            self.__writecheckfile(gR,self.params.root + '--'+ str(i) + '--' + str(self.params.smiles[i]))
        return gRs,gPs
    
    def _singlecompgraphcreation(self,info,unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP):
        u,v = self.__grabuv(info,unames,vnames)
        gR,gP = self._dglgraphcreation(u,v)
        if 'reaction' in self.params.graph:
            gR,gP = self._dglreactiongraphfcn(info,gR,gP,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP)
        elif 'molecular' in self.params.graph:
            gR = self._dglmoleculargraphfcn(info,gR,atomFRnames,bondFRnames,atomgeomnamesR)
        self.__writecheckfile(gR,self.params.root + '--'+ str(self.params.smiles))
        return gR,gP

    def _creategraphdglfrominfo(self,info):
        unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP = self.__getallnames()
        if isinstance(self.params.smiles,list):
            gRs,gPs = self._multicompgraphcreation(info,unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP)
            return gRs,gPs
        else:
            gR,gP = self._singlecompgraphcreation(info,unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP)
            return gR,gP

    def _creategraphdglfrominfolist(self,infolist):
        unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP = self.__getallnames()
        gRmat = []
        gPmat = []
        for info in infolist:
            if isinstance(self.params.smiles,list):
                gRs,gPs = self._multicompgraphcreation(info,unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP)
                gRmat.append(gRs)
                gPmat.append(gPs)
            else:
                gR,gP = self._singlecompgraphcreation(info,unames,vnames,atomFRnames,atomFPnames,bondFRnames,bondFPnames,atomgeomnamesR,atomgeomnamesP)
                gRmat.append(gR)
                gPmat.append(gP)
        return gRmat,gPmat
    
    def _creategraphsdgl(self):
        if self.params.getradical == 'YARP':
            if self.featurizer.bond_mats > 1:
                gR,gP = self._creategraphdglfrominfolist(self.infolist)
            else:
                if self.params.conformer.nconfs > 1:
                    gR,gP = self._creategraphdglfrominfolist(self.infolist)
                else:
                    gR,gP = self._creategraphdglfrominfo(self.info)
        else:
            if self.params.conformer.nconfs > 1:
                gR,gP = self._creategraphdglfrominfolist(self.infolist)
            else:
                gR,gP = self._creategraphdglfrominfo(self.info)
        return gR,gP

    def __writecheckfile(self,gR,index):
        if gR.ndata['x'].max() > 50: 
            with open('check.txt','a') as ff:
                ff.write('{}\n'.format(self.params.root + '--'+index))

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

    def _sampleindices(self,index,infolistusage=False):
        if infolistusage:
            return [self.info['Indices']]
        else:
            return [[info['Indices'] for info in self.infolist]]
    
    def __samplefunction(self,keycolumn,infolistusage=False):
        if infolistusage:
            if isinstance(self.params.smiles,list):
                return [self.info[keycolumn]]
            else:
                return [[info[keycolumn] for info in self.infolist]]
        else:
            if isinstance(self.params.smiles,str):
                return [self.info[keycolumn]]
            else:
                return [[value for key, value in self.info.items() if keycolumn in key]]

    def _samplereactiontype(self,infolistusage=False):
        return self.__samplefunction('rxntype',infolistusage)

    def _samplesmilesorinchi(self,smikey = 'Rsmiles',infolistusage=False):
        return self.__samplefunction(smikey,infolistusage)
    
    def __iteratethroughkeys(self,keys,infolistusage=False):
        result = [] 
        for key in keys: result += self.__samplefunction(keys,infolistusage)
        return result

    def _smilesandinchisampler(self,infolistusage=False):
        if 'reaction' in self.params.graph:
            keylist = ['Rsmiles','Psmiles','Rinchi','Pinchi']
        elif 'molecular' in self.params.graph:
            keylist = ['Rsmiles','Rinchi']
        return self.__iteratethroughkeys(keylist,infolistusage)

    def _getnumerictensorforinfo(self,vector,info):
        if vector is not None: 
            # Add what the target tensor is
            #targettensor = [gR.number_of_nodes(),gR.number_of_edges()]
            atensor = []
            
            if isinstance(vector,list) or isinstance(vector,omegaconf.listconfig.ListConfig):
                atensor += [float(info[output]) for output in vector]
            else:
                atensor += [float(info[vector])]
            
            atensor = torch.Tensor(atensor)
            return [atensor]
    
    def _sampletargets(self,infolistusage=False):
        if infolistusage:
            return [[self._getnumerictensorforinfo(self.params.target,info)] for info in self.infolist]
        else:
            return self._getnumerictensorforinfo(self.params.target,self.info)
    
    def _sampleadditionals(self,infolistusage=False):
        if infolistusage:
            return [[self._getnumerictensorforinfo(self.params.additional,info)] for info in self.infolist]
        else:
            return self._getnumerictensorforinfo(self.params.additional,self.info)

    def _generateaddonfeatureonthefly(self,addon,smiles):
        if addon in self.threed:
            addon = addon.split('-')[0]
            return GeometricFingerprint().smiles_to_fp([smiles], addon)
        elif addon in self.twod:
            return Fingerprint().smiles_to_fp([smiles], addon)
        elif addon in self.mordred_descriptors:
            mols = [Chem.MolFromSmiles(smiles)]
            calc = Calculator(descriptors, ignore_3D=False)
            df = calc.pandas(mols)
            return df[addon].values[0]
        elif addon in self.rdkit_descriptors:
            calc = RDKitDescriptors()
            return calc.compute(smiles, addon)
        elif addon in ['drfp','rxnfp']:
            generator =  ReactionFingerprint()
            return generator.smiles_to_fp([smiles], addon)
        elif addon in self.qm_descs:
            return None
        elif addon in self.other_descs:
            calc = Descriptors(calc=addon,is3d=True)
            return calc.Compute(smiles)
        elif addon in self.medchem_descs:
            return None
    
    def _molecularaddonvector(self,info,addon,smiles,prequel='R_',sequel=None):
        addonname = f'{prequel}addon_{addon}'
        if sequel is not None:
            addonname = f'{addonname}_{sequel}'
        info[addonname] = self._generateaddonfeatureonthefly(addon,smiles)
        return info

    def _createaddonintoinfo(self,info,addon,reactant,product=None,reaction=False,sequel=None):
        if reaction:
            if addon != ['drfp','rxnfp']:
                info = self._molecularaddonvector(info,addon,reactant,prequel='R_',sequel=sequel)
                info = self._molecularaddonvector(info,addon,product,prequel='P_',sequel=sequel)
            else:
                info = self._molecularaddonvector(info,addon,reactant,prequel='Rxn_',sequel=sequel)
        else:
            info = self._molecularaddonvector(info,addon,reactant,prequel='R_',sequel=sequel)

    def __getaddonnames(self,info,reaction=False,sequel=None):
        if reaction:
            if sequel is not None:
                addonnameR = f'R_Addon_{sequel}'
                addonnameP = f'P_Addon_{sequel}'
                info[addonnameR] = []
                info[addonnameP] = []
            else:
                addonnameR = 'R_Addon'
                addonnameP = 'P_Addon'
                info[addonnameR] = []
                info[addonnameP] = []
        else:
            if sequel is not None:
                addonnameR = f'R_Addon_{sequel}'
                info[addonnameR] = []
            else:
                addonnameR = 'R_Addon'
                info[addonnameR] = []
        return info
    
    def __smilesandaddonsarestrings(self,info):
        if 'reaction' in self.params.graph:
            info = self._createaddonintoinfo(info,self.params.addons,info['Rsmiles'],info['Psmiles'],reaction=True)
            info['R_Addon'] = info[f'R_Addon_{self.params.addons}']
            info['P_Addon'] = info[f'P_Addon_{self.params.addons}']
        elif 'molecular' in self.params.graph:
            info = self._createaddonintoinfo(info,self.params.addons,info['Rsmiles'])
            info['R_Addon'] = info[f'R_Addon_{self.params.addons}']
        return info
    
    def __addonsarestringabutsmilesisnot(self,info):
        if 'reaction' in self.params.graph:
            info['R_Addon'] = []
            info['P_Addon'] = []
            for addon in self.params.addons:
                info = self._createaddonintoinfo(info,addon,info['Rsmiles'],info['Psmiles'],reaction=True)
                info['R_Addon'] += info[f'R_Addon_{addon}']
                info['P_Addon'] += info[f'P_Addon_{addon}']
        elif 'molecular' in self.params.graph:
            info['R_Addon'] = []
            for addon in self.params.addons:
                info = self._createaddonintoinfo(info,addon,info['Rsmiles'])
                info['R_Addon'] += info[f'R_Addon_{addon}']
        return info
    
    def __smilesisalistbutaddonsarestring(self,info):
        if 'reaction' in self.params.graph:
            info[f'R_Addon'] = []
            info[f'P_Addon'] = []
            for smi in self.params.smiles:
                info[f'R_Addon_{smi}'] = []
                info[f'P_Addon_{smi}'] = []
                info = self._createaddonintoinfo(info,self.params.addons,info['Rsmiles'],info['Psmiles'],reaction=True,sequel=smi)
                info[f'R_Addon_{smi}'] = info[f'R_Addon_{self.params.addons}_{smi}']
                info[f'P_Addon_{smi}'] = info[f'P_Addon_{self.params.addons}_{smi}']
                info[f'R_Addon'] += info[f'R_Addon_{smi}']
                info[f'P_Addon'] += info[f'P_Addon_{smi}']
        elif 'molecular' in self.params.graph:
            info[f'R_Addon'] = []
            for smi in self.params.smiles:
                info = self._createaddonintoinfo(info,self.params.addons,info['Rsmiles'],sequel=smi)
                info[f'R_Addon_{smi}'] = info[f'R_Addon_{self.params.addons}_{smi}']
                info[f'R_Addon'] += info[f'R_Addon_{smi}']
    
    def __bothsmilesandaddonsarelist(self,info):
        if 'reaction' in self.params.graph:
            info[f'R_Addon'] = []
            info[f'P_Addon'] = []
            for smi in self.params.smiles:
                info[f'R_Addon_{smi}'] = []
                info[f'P_Addon_{smi}'] = []
                for addon in self.params.addons:
                    info = self._createaddonintoinfo(info,addon,info['Rsmiles'],info['Psmiles'],reaction=True,sequel=smi)
                    info[f'R_Addon_{smi}'] += info[f'R_Addon_{self.params.addons}_{smi}']
                    info[f'P_Addon_{smi}'] += info[f'P_Addon_{self.params.addons}_{smi}']
                info[f'R_Addon'] += info[f'R_Addon_{smi}']
                info[f'P_Addon'] += info[f'P_Addon_{smi}']
        elif 'molecular' in self.params.graph:
            info[f'R_Addon'] = []
            for addon in self.params.addons:
                info[f'R_Addon_{smi}'] = []
                for smi in self.params.smiles:
                    info = self._createaddonintoinfo(info,addon,info['Rsmiles'],sequel=smi)
                    info[f'R_Addon_{smi}'] += info[f'R_Addon_{self.params.addons}_{smi}']
                info[f'R_Addon'] += info[f'R_Addon_{smi}']
        return info

    def __grabaddons(self,info):
        if isinstance(self.params.smiles,str):
            if isinstance(self.params.addons,str):
                info = self.__smilesandaddonsarestrings(info)
            else:
                info = self.__addonsarestringabutsmilesisnot(info)
        else:
            if isinstance(self.params.addons,str):
                info = self.__smilesisalistbutaddonsarestring(info)
            else:
                info = self.__bothsmilesandaddonsarelist(info)
        return info
                
    def _combineaddonsintoinfovector(self,info):
        info = self.__getaddonnames(info)
        info = self.__grabaddons(info)
        return info

    def _combineaddonsininfolist(self,infolist):
        for info in infolist:
            info = self._combineaddonsintoinfovector(info)
        return infolist

    def obtainaddons(self):
        if self.params.getradical == 'YARP':
            if self.featurizer.bond_mats > 1:
                self.infolist = self._combineaddonsininfolist(self.infolist)
            else:
                if self.params.conformer.nconfs > 1:
                    self.infolist = self._combineaddonsininfolist(self.infolist)
                else:
                    self.info = self._combineaddonsintoinfovector(self.info)
        else:
            if self.params.conformer.nconfs > 1:
                self.infolist = self._combineaddonsininfolist(self.infolist)
            else:
                self.info = self._combineaddonsintoinfovector(self.info)

    def _tensorizeaddonsinfo(self):
        if self.params.multicomponent:
            if 'reaction' in self.params.graph:
                Routput = [] 
                Poutput = []
                for smi in self.params.smiles:
                    Routput += [torch.Tensor(self.info[f'R_Addon_{smi}'])]
                    Poutput += [torch.Tensor(self.info[f'P_Addon_{smi}'])]
                return [[Routput],[Poutput]]
            elif 'molecular' in self.params.graph:
                Routput = [] 
                for smi in self.params.smiles:
                    Routput += [torch.Tensor(self.info[f'R_Addon_{smi}'])]
                return [[Routput]]
        else:
            if 'reaction' in self.params.graph:
                return [torch.Tensor(self.info['R_Addon']),torch.Tensor(self.info['P_Addon'])]
            elif 'molecular' in self.params.graph:
                return [torch.Tensor(self.info['R_Addon'])]
    
    def _tensorizeaddonsinfolist(self):
        if self.params.multicomponent:
            if 'reaction' in self.params.graph:
                Routput = [] 
                Poutput = []
                for smi in self.params.smiles:
                    Routput += [torch.Tensor([info[f'R_Addon_{smi}'] for info in self.infolist])]
                    Poutput += [torch.Tensor([info[f'P_Addon_{smi}'] for info in self.infolist])]
                return [[Routput],[Poutput]]
            elif 'molecular' in self.params.graph:
                Routput = [] 
                for smi in self.params.smiles:
                    Routput += [torch.Tensor([info[f'R_Addon_{smi}'] for info in self.infolist])]
                return [[Routput]]
        else:
            if 'reaction' in self.params.graph:
                return [torch.Tensor([info[f'R_Addon'] for info in self.infolist]),torch.Tensor([info[f'P_Addon'] for info in self.infolist])]
            elif 'molecular' in self.params.graph:
                return [torch.Tensor([info[f'R_Addon'] for info in self.infolist])]

    def tensorizeinfo(self):
        if self.params.getradical == 'YARP':
            if self.featurizer.bond_mats > 1:
                samples = self._tensorizeaddonsinfolist()
            else:
                if self.params.conformer.nconfs > 1:
                    samples = self._tensorizeaddonsinfolist()
                else:
                    samples = self._tensorizeaddonsinfo()
        else:
            if self.params.conformer.nconfs > 1:
                samples = self._tensorizeaddonsinfolist()
            else:
                samples = self._tensorizeaddonsinfo()
        return samples


class DatasetCommands(DatasetSetups,BaseDatasetCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.InitialSetup() 
    
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
        self._setnormstoinfo(rxn,self.params.target,self.target_normalizer)
    
    def AddAdditionals(self,rxn):
        self._setnormstoinfo(rxn,self.params.additional,self.additional_normalizer)
    
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
            if self.params.conformer.nconfs > 1:
                self._multigeomyarpcase()

    def InfoorInfoList(self,Rind):
        self.info['Indices'] = Rind
        rxn = self.GetRow(Rind)
        self.AddTargets(rxn)
        self.AddAdditionals(rxn)
        self.AddINCHI(rxn)
        self.ChooseFeaturizer(rxn)
        self.AddFeatures(rxn)

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
        self.InfoorInfoList(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        self.samples = self.AddGraphsToSample(self.samples)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples
    
    def FingerprintModelSampler(self,index):
        infolistusage = self.__useinfolist()
        self.InfoorInfoList(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples

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