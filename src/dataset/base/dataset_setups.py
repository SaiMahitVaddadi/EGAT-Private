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
from omegaconf import OmegaConf

@dataclass
class DatasetParams:
    data_path: str
    rootfile: Union[str, List[str], Dict]
    smiles: Union[str, List[str]]
    target: Union[str, List[str]]
    additional: Optional[Union[str, List[str]]] = None
    target_normalizer: Optional[str] = None
    target_normalizer_params: Optional[Dict] = field(default_factory=dict)
    additional_normalizer: Optional[str] = None
    additional_normalizer_params: Optional[Dict] = field(default_factory=dict)
    normalize_step: str = "all"
    split: str = "train"
    fold: Optional[int] = None
    randomize: bool = True
    size: Optional[int] = None
    cache_size: int = 100
    denotation: Dict = field(default_factory=lambda: {"split_type": "random", "splittotrain": 0.8})
    exclude: Optional[List] = field(default_factory=list)
    n_splits: Optional[int] = None

class DatasetSetups(Dataset):
    def __init__(self, arguments):
        self.params = arguments
        self.SetupInits()
        
    def _checkifsmilescolumnislist(self):
        if isinstance(self.params.smiles,str): return False
        elif isinstance(self.params.smiles,list): return True
        else: return False

    def _obtainmordred(self):
        mols = [Chem.MolFromSmiles(smi) for smi in ['c1ccccc1Cl', 'c1ccccc1O', 'c1ccccc1N']]
        calc = Calculator(descriptors, ignore_3D=False)
        df = calc.pandas(mols)
        return df.columns.tolist()

    def SetupInits(self):
        self.catfile = os.path.join(self.root, 'Rxntype.txt')
        self.cat = []
        self.cache = {}
        self.cache_size = self.params.cache_size
        self.SetupAddOnKeys()

    def SetupAddOnKeys(self):
        self.twod = ['ecfp', 'bcut', 'rdkit', 'maccs', 'pubchem', 'layered', 'estate', 'torsional', 'avalon', '2dpharm', 'mapchiral', 
                     'mxfp', 'autocorr', 'erg', 'crippen', 'roth', 'laggner', 'lingo', 'map4', 'mhfp', 'morse', 'mordred', 'mqns', 
                     'patternfp', 'physchemfp', 'secfp', 'vsa', 'whim', 'cdkstandard', 'cdkextended', 'cdkgraph', 'hybridization', 
                     'cdkshortestpath', 'cdksubstructures', 'circular', 'cdkatompairs', 'babelfp', 'spectrophore', 'mol2vec', 'chemgpt12b', 
                     'chemgpt19m', 'chemgpt47m', 'molt5', 'robertazincllm', 'chembertmtrllm', 'chembertmlmllm', 'gpt2llm', 'scaffoldkeys', 
                     'cats', 'catsrdkit', 'gobbipharm', 'pmapper', 'chemblgin', 'jtvaezinc', 'chemblginedgepred', 'chemblgininfomax', 
                     'graphormerpcqm', 'chemblgincontext', 'hftransformer']
        self.threed = ['E3FP', 'RDF', 'GETAWAY', 'USR', 'USRCAT', 'WHIM','3dpharm','mordred-3d','mxfp-3d']
        self.rdkit_descriptors = [
            "Qed",
            "MolecularWeight",
            "GraphLength",
            "NumAtomStereoCenters",
            "HBondAcceptors",
            "HBondDonors",
            "NumRotBond",
            "Csp3",
            "numsp",
            "numsp2",
            "numsp3",
            "NumHeavyAtoms",
            "NumHeteroAtoms",
            "NumRings",
            "NumAromaticRings",
            "NumAliphaticRings",
            "SlogP",
            'tpsa',
            'pmi',
            'sascore',
            'scscore',
            'dockstring',
            'molcomplexity',
            'Richness','FG','RS','BM','IntDiv','HamDiv','Diam','SumDiam','SumDiv','Bot','SumBot','DPP','NCircles'
        ]
        self.mordred_descriptors = self._obtainmordred()
        self.qm_descs = [] #To update soon
        self.other_descs = ['molskill','descriptasourus-normed','descriptasourus','rascore-nn','rascore-xgb','syba','qep']
        self.medchem_descs = []
    
    def InitializeFolder(self):
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
    
    def ReadExcludeFile(self):
        self.exclude = []
        if os.path.exists(os.path.join(self.params.data_path, 'exclude.txt')):
            with open(os.path.join(self.params.data_path, 'exclude.txt'), 'r') as file:
                self.exclude = [line.strip() for line in file.readlines()]
        elif isinstance(self.params.exclude, list):
            self.exclude = self.params.exclude
    
    def FilterData(self):
        if len(self.exclude) != 0: 
            self.data = self.data[~self.data.index.isin(self.exclude)]
        #Check if the data is split
        if 'rxntype' not in self.data.columns:
            if not self._checkifsmilescolumnislist():
                datafilter = DenoteInputData(self.params.denotation.split_type,self.params.splittotrain,self.params.smiles,self.params.fold)
                self.data['rxntype'] = self.data[self.params.smiles].apply(datafilter.getctype,args=(self.molecular,))
            else:
                for smicolumn in self.params.smiles:
                    datafilter = DenoteInputData(self.params.denotation.split_type,self.params.splittotrain,smicolumn,self.params.fold)
                    self.data[f'rxntype_{smicolumn}'] = self.data[smicolumn].apply(datafilter.getctype,args=(self.molecular,))
        datafilter.GrabData()
        self.data = datafilter.data    
    

    def _splitcreator(self):
        self.splitter = Splitter(self.params)
        if self.params.n_splits is None: self.data = self.splitter.onefold(self.data)
        else: self.data = self.splitter.xfold(self.data)

    def _obtaindesiredsplit(self):
        if self.params.split in ['train','val','test']:
            if self.params.fold is None:
                return self.data[self.data['split'] == self.params.split]
            else:
                return self.data[self.data[f'fold{self.params.fold}_split'] == self.params.split]
        elif self.params.split in ['trainval','traintest','valtest']:
            if self.params.fold is None:
                return self.data[self.data['split'].isin(self.params.split)]
            else:
                return self.data[self.data[f'fold{self.params.fold}_split'].isin(self.params.split)]

    def CreateSplit(self):
        if 'split' not in self.data.columns:
            self._splitcreator()
            self._obtaindesiredsplit()
        else:
            self._obtaindesiredsplit()

    def CreateDataset(self):    
        if self.params.randomize: self.data = self.data.sample(frac=1)
        if self.params.size is not None: self.data = self.data.iloc[:self.params.size,:]

    def _datatofit(self,columns):
        if isinstance(columns,list) or isinstance(columns,omegaconf.listconfig.ListConfig):
            data_to_fit = self.data[columns].values
        else:
            data_to_fit = self.data[columns].values.reshape(-1,1)
        return data_to_fit

    def SetupTargetNormalizer(self):
        if self.params.target_normalizer is not None:
            self.target_normalizer = DataNormalizer(self.params.target_normalizer,**self.params.target_normalizer_params)
            data_to_fit = self._datatofit(self.params.target)
            self.target_normalizer.fit(data_to_fit)
    
    def SetupAdditionalNormalizer(self):
        if self.params.additional_normalizer is not None:
            self.additional_normalizer = DataNormalizer(self.params.additional_normalizer,**self.params.additional_normalizer_params)
            data_to_fit = self._datatofit(self.params.additional)
            self.additional_normalizer.fit(data_to_fit)

    def InitialSetup(self):
        self.InitializeFolder()
        self.ReadFile()
        self.ReadExcludeFile()
        if self.params.normalize_step == 'all':
            self.SetupTargetNormalizer()
            self.SetupAdditionalNormalizer()
        self.CreateSplit()
        self.FilterData()
        self.CreateDataset()
        if self.params.normalize_step == 'subset':
            self.SetupTargetNormalizer()
            self.SetupAdditionalNormalizer()
        self.info = {}
    
if __name__ == "__main__":

    # Example configuration
    config = OmegaConf.create({
        "data_path": "./data",
        "rootfile": "/Users/svaddadi/Documents/GitHub/EGAT/tests/dataset/example.csv",
        "smiles": "smiles",
        "target": "target",
        "additional": None,
        "target_normalizer": "minmax",
        "target_normalizer_params": {},
        "additional_normalizer": None,
        "additional_normalizer_params": {},
        "normalize_step": "all",
        "split": "train",
        "fold": None,
        "randomize": True,
        "size": None,
        "cache_size": 100,
        "denotation": {
            "split_type": "random",
            "splittotrain": 0.8
        },
        "exclude": [],
        "n_splits": None
    })

    # Create an instance of DatasetSetups
    dataset = DatasetSetups(config)

    # Perform initial setup
    dataset.InitialSetup()

    # Access the processed data
    print("Processed Data:")
    print(dataset.data.head())