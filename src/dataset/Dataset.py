
import os 
import pandas as pd 
import torch
import traceback
import dgl 
from torch.utils.data import Dataset
from rdkit import Chem
import polars as pl
from pyspark.sql import Row
import toml 
import spark 
import h5py
import omegaconf

from ..splitting.split import Splitter
from ..utils.descriptors.egat.encodings import Encodings
from ..utils.database.csvfunctions import DenoteInputData
from ..utils.descriptors.fingerprint.threedimensional import GeometricFingerprint
from ..utils.descriptors.fingerprint.twodimensional import Fingerprint
from ..graph.molecular.Molecule import MoleculeFeaturizer
from ..graph.molecular.MoleculeGeometry import MoleculeFeaturizerwithGeometry
from ..graph.reaction.ReactionGeometry import ReactionFeaturizerwithGeometry
from ..graph.molecular.MoleculeGlobalGeometry import MoleculeFeaturizerwithPaddingandGeometry
from ..graph.reaction.ReactionGlobalGeometry import ReactionFeaturizerwithPaddingandGeometry
from ..graph.molecular.MoleculeGlobal import MoleculeFeaturizerwithPadding
from ..graph.reaction.ReactionGlobal import ReactionFeaturizerwithPadding
from ..graph.reaction.Reaction import ReactionFeaturizer
from ..tools.jepa.tools import mask_uv_vectors,mask_node_features,mask_node_features_v2,mask_edge_features,mask_edge_features_v2,mask_node_and_edges,mask_node_and_edges_v2,mask_node_and_edges_v3,mask_node_and_neighbors
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class DenotationParams:
    split_type: str
    splittotrain: str
    smilescolumn: str
    fold: Optional[int] = None


@dataclass
class EGATDatasetParams:
    data_path: str
    rootfile: str
    exclude: Optional[List[str]] = None
    denotation: Optional[DenotationParams] = None
    splittotrain: Optional[str] = None
    smilescolumn: Optional[str] = None
    fold: Optional[int] = None
    target: Union[str, List[str]] = None
    additionals: Optional[Union[str, List[str]]] = None
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



class EGATDataset(Dataset):
    def __init__(self,arguments):
        self.params = arguments
        self.catfile = os.path.join(self.root, 'Rxntype.txt')
        self.cat = []
        self.cache = {}
        self.cache_size = self.params.cache_size
        self.twod = ['ecfp', 'bcut', 'rdkit', 'maccs', 'pubchem', 'layered', 'estate', 'torsional', 'avalon', '2dpharm', 'mapchiral', 
                     'mxfp', 'autocorr', 'erg', 'crippen', 'roth', 'laggner', 'lingo', 'map4', 'mhfp', 'morse', 'mordred', 'mqns', 
                     'patternfp', 'physchemfp', 'secfp', 'vsa', 'whim', 'cdkstandard', 'cdkextended', 'cdkgraph', 'hybridization', 
                     'cdkshortestpath', 'cdksubstructures', 'circular', 'cdkatompairs', 'babelfp', 'spectrophore', 'mol2vec', 'chemgpt12b', 
                     'chemgpt19m', 'chemgpt47m', 'molt5', 'robertazincllm', 'chembertmtrllm', 'chembertmlmllm', 'gpt2llm', 'scaffoldkeys', 
                     'cats', 'catsrdkit', 'gobbipharm', 'pmapper', 'chemblgin', 'jtvaezinc', 'chemblginedgepred', 'chemblgininfomax', 
                     'graphormerpcqm', 'chemblgincontext', 'hftransformer']

        self.Initialize()
        self.ReadFile()
        self.ReadExclude()
        self.CreateSplit()
        self.FilterData()
        self.CreateDataset()

        
    def Initialize(self):
        if not os.path.isdir(self.params.data_path): os.mkdir(self.params.data_path)
        
    def ReadFile(self):
        if '.csv' in self.params.rootfile:
            self.data = pd.read_csv(self.params.rootfile)
        elif '.tsv' in self.params.rootfile:
            self.data = pd.read_tsv(self.params.rootfile)
        elif '.xlsx' in self.params.rootfile:
            self.data = pd.read_excel(self.params.rootfile)
        elif '.txt' in self.params.rootfile:
            self.data = pd.read_csv(self.params.rootfile, delimiter='\t')
        elif isinstance(self.params.rootfile, list):
            if '.csv' in self.params.rootfile[0]:
                self.data = pd.concat([pd.read_csv(file) for file in self.params.rootfile], ignore_index=True)
            else:
                self.data = pd.DataFrame({'smiles': self.params.rootfile})
        elif isinstance(self.params.rootfile, dict):
            self.data = pd.DataFrame(self.params.rootfile)
    
    def ReadExclude(self):
        self.exclude = []
        if os.path.exists(os.path.join(self.params.data_path, 'exclude.txt')):
            with open(os.path.join(self.params.data_path, 'exclude.txt'), 'r') as file:
                self.exclude = [line.strip() for line in file.readlines()]
        elif isinstance(self.params.exclude, list):
            self.exclude = self.params.exclude
    
    def CreateSplit(self):
        if 'split' not in self.data.columns:
            self.splitter = Splitter(self.params)
            if self.params.n_splits is None: self.data = self.splitter.onefold(self.data)
            else: self.data = self.splitter.xfold(self.data)
        else:
            self.data = self.data


    def FilterData(self):
        if len(self.exclude) != 0: 
            self.data = self.data[~self.data.index.isin(self.exclude)]
        
        datafilter = DenoteInputData(self.params.denotation.split_type,self.params.splittotrain,self.params.smilescolumn,self.params.fold)
        #Check if the data is split
        if 'rxntype' not in self.data.columns:
            self.data['rxntype'] = self.data[self.smiles].apply(datafilter.getctype,args=(self.molecular,))

        datafilter.GrabData()
        self.data = datafilter.data    
    
    def CreateDataset(self):    
        if self.params.randomize: self.data = self.data.sample(frac=1)
        if self.params.size is not None: self.data = self.data.iloc[:self.params.size,:]


    def GetRow(self,Rind):
        if isinstance(Rind,str):
            rxn = self.data.loc[Rind,:]
        else:
            rxn = self.data.iloc[Rind,:]
        return rxn

    def AddTargets(self,rxn):
        ###### SAVE THE TARGET AND ADDTIONAL DATA TO THE DICTIONARY
        if isinstance(self.params.target,list) or isinstance(self.params.target,omegaconf.listconfig.ListConfig):
            for outputs in self.params.target:
                self.info[outputs] = rxn[outputs]
        else:
            self.info[self.params.target] = rxn[self.params.target]
    
    def AddAdditionals(self,rxn):
        adddict = dict()
        if isinstance(self.params.additional,list)or isinstance(self.params.additional,omegaconf.listconfig.ListConfig):
            for outputs in self.params.additional:
                self.info[outputs] = rxn[outputs]
        elif self.params.additional is not None:
            self.info[self.params.additional] = rxn[self.params.additional]
            
    def AddINCHI(self,rxn):
        if 'reaction' in self.params.graph:
            RPsmiles = rxn[self.smiles].split('>>')
            Rsmiles = RPsmiles[0]
            Psmiles = RPsmiles[1]        
            self.info['Rinchi'] = getInchifromSMILES(Rsmiles)
            self.info['Pinchi'] = getInchifromSMILES(Psmiles)
            NRsmiles = RemoveMapping(Rsmiles)
            NPsmiles = RemoveMapping(Psmiles)
            self.info["Rsmiles"] = Chem.MolToSmiles(NRsmiles)
            self.info["Psmiles"] = Chem.MolToSmiles(NPsmiles)
        elif 'molecular' in self.params.graph:
            Rsmiles = rxn[self.smiles]
            self.info['Rinchi'] = getInchifromSMILES(Rsmiles)
            NRsmiles = RemoveMapping(Rsmiles)
            self.info["Rsmiles"] = NRsmiles


    def AddFeatures(self,rxn,info):
        if 'molecular' in self.params.graph:
            if self.params.dimension == '2d':
                if self.params.addmissingbonds in ['global','hbonds','global+hbonds']:
                    self.featurizer = MoleculeFeaturizerwithPadding(rxn[self.smiles],self.params)
                else:
                    self.featurizer = MoleculeFeaturizer(rxn[self.smiles],self.params)
            elif self.params.dimension == '3d':
                if self.params.addmissingbonds in ['global','hbonds','global+hbonds']:
                    self.featurizer = MoleculeFeaturizerwithPaddingandGeometry(rxn[self.smiles],self.params)
                else:
                    self.featurizer = MoleculeFeaturizerwithGeometry(rxn[self.smiles],self.params)
        if 'reaction' in self.params.graph:
            if self.params.dimension == '2d':
                if self.params.addmissingbonds in ['global','hbonds','global+hbonds']:
                    self.featurizer = ReactionFeaturizerwithPadding(rxn[self.smiles],self.params)
                else:
                    self.featurizer = ReactionFeaturizer(rxn[self.smiles],self.params)
            elif self.params.dimension == '3d':
                if self.params.addmissingbonds in ['global','hbonds','global+hbonds']:
                    self.featurizer = ReactionFeaturizerwithPaddingandGeometry(rxn[self.smiles],self.params)
                else:
                    self.featurizer = ReactionFeaturizerwithGeometry(rxn[self.smiles],self.params)

        
        if 'reaction' in self.params.graph:
            self.info['u'] = self.featurizer.reactant.edges_u
            self.info['v'] = self.featurizer.reactant.edges_v
            self.info['atom_F_R'] = self.featurizer.reactant.atom_features
            self.info['bond_F_R'] = self.featurizer.reactant.bond_features
            self.info['atom_F_P'] = self.featurizer.product.atom_features
            self.info['bond_F_P'] = self.featurizer.product.bond_features
        elif 'molecular' in self.params.graph:
            self.info['u'] = self.featurizer.edges_u
            self.info['v'] = self.featurizer.edges_v
            self.info['atom_F_R'] = self.featurizer.atom_features
            self.info['bond_F_R'] = self.featurizer.bond_features
            



    def Convert(self,Rind):
        pt = Chem.GetPeriodicTable()
        num2element = {value: key.capitalize() for key, value in Encodings().el_to_an.items()}
        
        self.info = {}
        try:
            self.info['Indices'] = Rind
            rxn = self.GetRow(Rind)
            self.AddTargets(rxn)
            self.AddAdditionals(rxn)
            self.AddINCHI(rxn)
            self.AddFeatures(rxn)
        except:
            print(self.root + '--'+ str(Rind) + ' failed')
            print(traceback.print_exc())
            return None
        

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


    def CreateGraphDGL(self,index):
        u, v = torch.Tensor(self.info['u']).int(),torch.Tensor(self.info['v']).int()
        if self.params.jepa: u,v = mask_uv_vectors(u,v)
        gR   = dgl.graph((u,v))
        gP   = dgl.graph((u,v))

        if 'reaction' in self.params.graph:
            gR.ndata['x'] = torch.Tensor(self.info['atom_F_R'])
            gR.edata['x'] = torch.Tensor(self.info['bond_F_R'])
            gP.ndata['x'] = torch.Tensor(self.info['atom_F_P'])
            gP.edata['x'] = torch.Tensor(self.info['bond_F_P'])
            if self.params.jepa: chosen_index = self.RunJEPA(gR)
            if self.params.jepa: chosen_index = self.RunJEPA(gP)
                

        elif 'molecular' in self.params.graph:
            gR.ndata['x'] = torch.Tensor(self.info['atom_F_R'])
            gR.edata['x'] = torch.Tensor(self.info['bond_F_R'])
            if self.params.jepa: chosen_index = self.RunJEPA(gR)
                
        
                
        if gR.ndata['x'].max() > 50: 
            with open('check.txt','a') as ff:
                ff.write('{}\n'.format(self.params.root + '--'+index))
        if 'reaction' in self.params.graph:
            return gR,gP
        elif 'molecular' in self.params.graph:
            return gR
        
    def CreateGraphPyG(self, index):

        u, v = torch.tensor(self.info['u'], dtype=torch.long), torch.tensor(self.info['v'], dtype=torch.long)
        if self.params.jepa:
            u, v = mask_uv_vectors(u, v)
        
        edge_index = torch.stack([u, v], dim=0)
        
        if 'reaction' in self.params.graph:
            x_R = torch.tensor(self.info['atom_F_R'], dtype=torch.float)
            edge_attr_R = torch.tensor(self.info['bond_F_R'], dtype=torch.float)
            x_P = torch.tensor(self.info['atom_F_P'], dtype=torch.float)
            edge_attr_P = torch.tensor(self.info['bond_F_P'], dtype=torch.float)
            
            gR = Data(x=x_R, edge_index=edge_index, edge_attr=edge_attr_R)
            gP = Data(x=x_P, edge_index=edge_index, edge_attr=edge_attr_P)
            
            if self.params.jepa: chosen_index = self.RunJEPA(gR)
            if self.params.jepa: chosen_index = self.RunJEPA(gP)



            return gR, gP
        
        elif 'molecular' in self.params.graph:
            x_R = torch.tensor(self.info['atom_F_R'], dtype=torch.float)
            edge_attr_R = torch.tensor(self.info['bond_F_R'], dtype=torch.float)
            
            gR = Data(x=x_R, edge_index=edge_index, edge_attr=edge_attr_R)
            
            if self.params.jepa: chosen_index = self.RunJEPA(gR)
            
            return gR


    def CreateGraph(self, index):
        if self.params.mode == 'dgl':
            if 'reaction' in self.params.graph:
                gR, gP = self.CreateGraphDGL(index)
                return gR, gP
            elif 'molecular' in self.params.graph:
                gR = self.CreateGraphDGL(index)
                return gR
        elif self.params.mode == 'pyg':
            if 'reaction' in self.params.graph:
                gR, gP = self.CreateGraphPyG(index)
                return gR, gP
            elif 'molecular' in self.params.graph:
                gR = self.CreateGraphPyG(index)
                return gR


    def CreateSampler(self,gR,gP=None):

        samples = [self.info['Indices'],self.info['rxntype']]

        # Add the graph
        if 'reaction' in self.params.graph:
            samples += [gR,gP]
            samples += [[self.info['Rsmiles'],self.info['Psmiles'],self.info['Rinchi'],self.info['Pinchi']]]
        elif 'molecular' in self.params.graph:
            samples += [gR]
            samples += [[self.info['Rsmiles'],self.info['Rinchi']]]
        
        samples += self.TargetTensor()
        samples += self.AdditionalTensor()
        samples += self.Addons()
        return samples



    def TargetTensor(self):
        # Add what the target tensor is
        #targettensor = [gR.number_of_nodes(),gR.number_of_edges()]
        targettensor = []
        
        if isinstance(self.params.target,list) or isinstance(self.params.target,omegaconf.listconfig.ListConfig):
            targettensor += [float(self.info[output]) for output in self.params.target]
        else:
            targettensor += [float(self.info[self.params.target])]
        
        targettensor = torch.Tensor(targettensor)
        return [targettensor]
    
    def AdditionalTensor(self):
        # Add what the Additional Tensor is if needed
        if self.additional is not None:
            additionaltensor = []
            if isinstance(self.additional,list) or isinstance(self.additional,omegaconf.listconfig.ListConfig):
                additionaltensor += [float(self.info[output]) for output in self.additional]
            else:
                additionaltensor += [float(self.info[self.additional])]
            additionaltensor = torch.Tensor(additionaltensor)
            return [additionaltensor]
    


    def AddonFunction(self,addon,smiles):
        if addon in ['E3FP', 'RDF', 'GETAWAY', 'USR', 'USRCAT', 'WHIM','3dpharm','mordred-3d','mxfp-3d']:
            addon = addon.split('-')[0]
            return GeometricFingerprint().smiles_to_fp([smiles], addon)
        elif addon in self.twod:
            return Fingerprint().smiles_to_fp([smiles], addon)
        elif addon in ['kelley-rdkit','kelley-rdkit-norm']:
            return None
        elif addon in ['volume','mw','tpsa','logp','qed']:
            return None
        elif addon in ['drfp','rxnfp']:
            return None
        
        
    def CreateAddons(self):
        if isinstance(self.params.addons, list):
            if 'reaction' in self.params.graph:
                    self.info['R_Addon'] = []
                    self.info['P_Addon'] = []
                    for addon in self.params.addons:
                        self.info['R_Addon'].append(self.AddonFunction(addon,self.info['Rsmiles']))
                        self.info['P_Addon'].append(self.AddonFunction(addon,self.info['Psmiles']))
            elif 'molecular' in self.params.graph:
                self.info['R_Addon'] = []
                for addon in self.params.addons:
                    self.info['R_Addon'].append(self.AddonFunction(addon,self.info['Rsmiles']))
        elif isinstance(self.params.addons, str):
            if 'reaction' in self.params.graph:
                self.info['R_Addon'] = self.AddonFunction(self.params.addons,self.info['Rsmiles'])
                self.info['P_Addon'] = self.AddonFunction(self.params.addons,self.info['Psmiles'])
            elif 'molecular' in self.params.graph:
                self.info['R_Addon'] = self.AddonFunction(self.params.addons,self.info['Rsmiles'])

    def Addons(self):
        # Add what the Add-Ons are
        if self.params.addons is not None: 
            self.CreateAddons()
            if 'reaction' in self.params.graph:
                return [torch.Tensor(self.info['R_Addon']),torch.Tensor(self.info['P_Addon'])]
            elif 'molecular' in self.params.graph:
                return [torch.Tensor(self.info['R_Addon'])]
                # save in cache
    
    

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                self.Convert(index)            
                if 'reaction' in self.params.graph:
                    gR,gP = self.CreateGraph(index)
                    samples = self.CreateSampler(gR,gP)
                elif 'molecular' in self.params.graph:
                    gR = self.CreateGraph(index)
                    samples = self.CreateSampler(gR)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
                
    
    