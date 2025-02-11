
import os 
import pandas as pd 
import traceback
import dgl 
from torch.utils.data import Dataset
from rdkit import Chem
import polars as pl
from pyspark.sql import Row
import dask.dataframe as dd
import torch 


from ...splitting.split import Splitter
from ...utils.descriptors.egat.encodings import Encodings
from ...utils.database.csvfunctions import DenoteInputData
from ...graph.Molecule import MoleculeFeaturizer
from ...graph.Reaction import ReactionFeaturizer
from ...graph.MoleculeGeometry import MoleculeFeaturizerwithGeometry
from ...graph.ReactionGeometry import ReactionFeaturizerwithGeometry
from ...graph.MoleculeGlobalGeometry import MoleculeFeaturizerwithPaddingandGeometry
from ...graph.ReactionGlobalGeometry import ReactionFeaturizerwithPaddingandGeometry
from ...graph.MoleculeGlobal import MoleculeFeaturizerwithPadding
from ...graph.ReactionGlobal import ReactionFeaturizerwithPadding
from ...tools.jepa.tools import mask_uv_vectors,mask_node_features,mask_node_features_v2
from dataclasses import dataclass, field
from typing import List, Optional, Union
from torch_geometric.data import Data
from tqdm import tqdm
import json
import toml


@dataclass
class DatasetParams:
    data_path: str
    rootfile: str
    class_choice: Optional[List[str]] = None
    cache_size: int = 100
    randomize: bool = False
    size: Optional[int] = None
    target: Optional[Union[str, List[str]]] = None
    additional: Optional[Union[str, List[str]]] = None
    exclude: Optional[List[str]] = None
    denotation: Optional[str] = None
    splittotrain: Optional[str] = None
    smilescolumn: Optional[str] = None
    modes: Optional[dict] = None
    dimension: str = '2d'
    addmissingbonds: Optional[str] = None
    jepa: bool = False
    addons: Optional[Union[str, List[str]]] = None

class JSONSaver:
    def __init__(self, arguments):
        self.params = arguments

    def __init__(self,arguments):
        self.params = arguments
        self.catfile = os.path.join(self.root, 'Rxntype.txt')
        self.cat = []
        self.cache = {}
        self.cache_size = self.params.cache_size
        
    def Initialize(self):
        if not os.path.isdir(self.params.data_path): os.mkdir(self.params.data_path)
        if not self.paramsclass_choice is None:
            self.cat = [k for k in self.cat if k in self.params.class_choice]

    def ReadFile(self):
        if '.csv' in self.params.rootfile:
            self.data = pd.read_csv(self.params.rootfile)
        elif '.tsv' in self.params.rootfile:
            self.data = pd.read_tsv(self.params.rootfile)
        elif '.xlsx' in self.params.rootfile:
            self.data = pd.read_excel(self.params.rootfile)
        elif '.txt' in self.params.rootfile:
            self.data = pd.read_csv(self.params.rootfile, delimiter='\t')
    
    def ReadExclude(self):
        self.exclude = []
        if os.path.exists(os.path.join(self.params.data_path, 'exclude.txt')):
            with open(os.path.join(self.params.data_path, 'exclude.txt'), 'r') as file:
                self.exclude = [line.strip() for line in file.readlines()]
        elif isinstance(self.params.exclude, list):
            self.exclude = self.params.exclude


    def FilterData(self):
        if len(self.exclude) != 0: 
            self.data = self.data[~self.data.index.isin(self.exclude)]
        
        datafilter = DenoteInputData(self.params.denotation.split_type,self.params.splittotrain,self.params.smilescolumn)
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

    def AddSplit(self, rxn):
        if 'split' in self.data.columns:
            self.info['split'] = rxn['split']
        else:
            self.info['split'] = 'all'
        
    def AddAdditionals(self,rxn):
        adddict = dict()
        if isinstance(self.params.additional,list)or isinstance(self.params.additional,omegaconf.listconfig.ListConfig):
            for outputs in self.params.additional:
                self.info[outputs] = rxn[outputs]
        elif self.params.additional is not None:
            self.info[self.params.additional] = rxn[self.params.additional]
            
    def AddINCHI(self,rxn):
        if 'reaction' in self.params.modes.graph:
            RPsmiles = rxn[self.smiles].split('>>')
            Rsmiles = RPsmiles[0]
            Psmiles = RPsmiles[1]        
            self.info['Rinchi'] = getInchifromSMILES(Rsmiles)
            self.info['Pinchi'] = getInchifromSMILES(Psmiles)
            NRsmiles = RemoveMapping(Rsmiles)
            NPsmiles = RemoveMapping(Psmiles)
            self.info["Rsmiles"] = Chem.MolToSmiles(NRsmiles)
            self.info["Psmiles"] = Chem.MolToSmiles(NPsmiles)
        elif 'molecular' in self.params.modes.graph:
            Rsmiles = rxn[self.smiles]
            self.info['Rinchi'] = getInchifromSMILES(Rsmiles)
            NRsmiles = RemoveMapping(Rsmiles)
            self.info["Rsmiles"] = NRsmiles


    def AddFeatures(self,rxn,info):
        if 'molecular' in self.params.modes.graph:
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
        if 'reaction' in self.params.modes.graph:
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

        
        if 'reaction' in self.params.modes.graph:
            self.info['u'] = self.featurizer.reactant.edges_u
            self.info['v'] = self.featurizer.reactant.edges_v
            self.info['atom_F_R'] = self.featurizer.reactant.atom_features
            self.info['bond_F_R'] = self.featurizer.reactant.bond_features
            self.info['atom_F_P'] = self.featurizer.product.atom_features
            self.info['bond_F_P'] = self.featurizer.product.bond_features
        elif 'molecular' in self.params.modes.graph:
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
            self.AddSplit(rxn)
        except:
            print(self.root + '--'+ str(Rind) + ' failed')
            print(traceback.print_exc())
            return None
        
    def SaveRowToTOML(self, index):
        try:
            self.Convert(index)
            rxntype = self.info.get('rxntype', 'unknown')
            rxntype_folder = os.path.join(self.params.data_path, rxntype)
            if not os.path.exists(rxntype_folder):
                os.makedirs(rxntype_folder)
            toml_file_path = os.path.join(rxntype_folder, f"{index}.toml")
            with open(toml_file_path, 'w') as toml_file:
                toml.dump(self.info, toml_file)
        except Exception as e:
            print(self.root + '--'+ str(index) + ' failed')
            print(traceback.print_exc())

    def SaveInfoToTOML(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to TOML files"):
            self.SaveRowToTOML(index)




class JSONDataset(Dataset):
    def __init__(self, arguments):
        self.params = arguments
    
    def LoadJSONDataFrame(self):
        try:
            self.data = pd.read_hdf(os.path.join(self.params.data_path, 'data.h5'), 'data')
        except Exception as e:
            print(f"Failed to load JSON DataFrame from {self.params.data_path}")
            print(traceback.print_exc())



    def ReadExclude(self):
        self.exclude = []
        if os.path.exists(os.path.join(self.params.data_path, 'exclude.txt')):
            with open(os.path.join(self.params.data_path, 'exclude.txt'), 'r') as file:
                self.exclude = [line.strip() for line in file.readlines()]
        elif isinstance(self.params.exclude, list):
            self.exclude = self.params.exclude


    def FilterData(self):
        if len(self.exclude) != 0: 
            self.data = self.data[~self.data.index.isin(self.exclude)]
        
        datafilter = DenoteInputData(self.params.denotation.split_type,self.params.splittotrain,self.params.smilescolumn)
        #Check if the data is split
        if 'rxntype' not in self.data.columns:
            self.data['rxntype'] = self.data[self.smiles].apply(datafilter.getctype,args=(self.molecular,))

        datafilter.GrabData()
        self.data = datafilter.data    
    
    def CreateDataset(self):    
        if self.params.randomize: self.data = self.data.sample(frac=1)
        if self.params.size is not None: self.data = self.data.iloc[:self.params.size,:]


    def GetRow(self, Rind):
        if isinstance(Rind, str):
            rxn = self.data.loc[Rind, :].to_dict()
        else:
            rxn = self.data.iloc[Rind, :].to_dict()
        self.info = rxn

    def CreateGraphDGL(self,index):
        u, v = torch.Tensor(self.info['u']).int(),torch.Tensor(self.info['v']).int()
        if self.params.jepa: u,v = mask_uv_vectors(u,v)
        gR   = dgl.graph((u,v))
        gP   = dgl.graph((u,v))

        if 'reaction' in self.params.modes.graph:
            gR.ndata['x'] = torch.Tensor(self.info['atom_F_R'])
            gR.edata['x'] = torch.Tensor(self.info['bond_F_R'])
            gP.ndata['x'] = torch.Tensor(self.info['atom_F_P'])
            gP.edata['x'] = torch.Tensor(self.info['bond_F_P'])
            chosen_index = mask_node_features(gR)
            chosen_indexP= mask_node_features(gP)
                

        elif 'molecular' in self.params.modes.graph:
            gR.ndata['x'] = torch.Tensor(self.info['atom_F_R'])
            gR.edata['x'] = torch.Tensor(self.info['bond_F_R'])
            chosen_index = mask_node_features(gR)
                
        
                
        if gR.ndata['x'].max() > 50: 
            with open('check.txt','a') as ff:
                ff.write('{}\n'.format(self.params.root + '--'+index))
        if 'reaction' in self.params.modes.graph:
            return gR,gP
        elif 'molecular' in self.params.modes.graph:
            return gR
        
    def CreateGraphPyG(self, index):

        u, v = torch.tensor(self.info['u'], dtype=torch.long), torch.tensor(self.info['v'], dtype=torch.long)
        if self.params.jepa:
            u, v = mask_uv_vectors(u, v)
        
        edge_index = torch.stack([u, v], dim=0)
        
        if 'reaction' in self.params.modes.graph:
            x_R = torch.tensor(self.info['atom_F_R'], dtype=torch.float)
            edge_attr_R = torch.tensor(self.info['bond_F_R'], dtype=torch.float)
            x_P = torch.tensor(self.info['atom_F_P'], dtype=torch.float)
            edge_attr_P = torch.tensor(self.info['bond_F_P'], dtype=torch.float)
            
            gR = Data(x=x_R, edge_index=edge_index, edge_attr=edge_attr_R)
            gP = Data(x=x_P, edge_index=edge_index, edge_attr=edge_attr_P)
            
            chosen_index_R = mask_node_features(gR)
            chosen_index_P = mask_node_features(gP)
            
            return gR, gP
        
        elif 'molecular' in self.params.modes.graph:
            x_R = torch.tensor(self.info['atom_F_R'], dtype=torch.float)
            edge_attr_R = torch.tensor(self.info['bond_F_R'], dtype=torch.float)
            
            gR = Data(x=x_R, edge_index=edge_index, edge_attr=edge_attr_R)
            
            chosen_index_R = mask_node_features(gR)
            
            return gR


    def CreateGraph(self, index):
        if self.params.mode == 'dgl':
            if 'reaction' in self.params.modes.graph:
                gR, gP = self.CreateGraphDGL(index)
                return gR, gP
            elif 'molecular' in self.params.modes.graph:
                gR = self.CreateGraphDGL(index)
                return gR
        elif self.params.mode == 'pyg':
            if 'reaction' in self.params.modes.graph:
                gR, gP = self.CreateGraphPyG(index)
                return gR, gP
            elif 'molecular' in self.params.modes.graph:
                gR = self.CreateGraphPyG(index)
                return gR


    def CreateSampler(self,gR,gP=None):

        samples = [self.info['Indices'],self.info['rxntype']]

        # Add the graph
        if 'reaction' in self.params.modes.graph:
            samples += [gR,gP]
            samples += [[self.info['Rsmiles'],self.info['Psmiles'],self.info['Rinchi'],self.info['Pinchi']]]
        elif 'molecular' in self.params.modes.graph:
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
    


    def Addons(self):
        # Add what the Add-Ons are
        if self.params.addons is not None: 
            if 'reaction' in self.params.modes.graph:
                return [torch.Tensor(self.info['R_Addon']),torch.Tensor(self.info['P_Addon'])]
            elif 'molecular' in self.params.modes.graph:
                return [torch.Tensor(self.info['R_Addon'])]
                    
                # save in cache
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                self.Convert(index)            
                if 'reaction' in self.params.modes.graph:
                    gR,gP = self.CreateGraph(index)
                    samples = self.CreateSampler(gR,gP)
                elif 'molecular' in self.params.modes.graph:
                    gR = self.CreateGraph(index)
                    samples = self.CreateSampler(gR)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None







 

