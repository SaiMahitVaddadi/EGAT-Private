import os 
import pandas as pd 
from torch.utils.data import Dataset
from ...splitting.split import Splitter
from ...utils.database.csvfunctions import DenoteInputData
from ...processing.normalize import DataNormalizer
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict

@dataclass
class BaseDatasetParams:
    data_path: str = ""
    rootfile: Union[str, List[str], Dict] = ""
    smiles: Union[str, List[str]] = ""
    target: Union[str, List[str]] = ""
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
    split_type: str = "random"
    train_size: float = 0.8
    random_state: int = 42
    denotation: Dict = field(default_factory=lambda: {"split_type": "random", "splittotrain": 0.8})
    exclude: Optional[List] = field(default_factory=list)
    n_splits: Optional[int] = None

class DatasetSetups(Dataset):
    def __init__(self, arguments,split=None):
        self.params = arguments
        self.split = split if split != None else 'all'
        self.SetupInits()
        self.checkmolecular()
        
    def _checkifsmilescolumnislist(self):
        if isinstance(self.params.smiles,str): return False
        elif isinstance(self.params.smiles,list): return True
        else: return False
    def SetupInits(self):
        self.root = self.params.data_path
        self.catfile = os.path.join(self.root, 'Rxntype.txt')
        self.cat = []
        self.cache = {}
        self.cache_size = self.params.cache_size
        
    def InitializeFolder(self):
        if not os.path.isdir(self.params.data_path): os.mkdir(self.params.data_path)

    def checkmolecular(self):
        if self.params.graph == 'reaction':
            self.molecular = False
        elif self.params.graph == 'molecular':
            self.molecular = True

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
        elif isinstance(self.params.rootfile, pd.DataFrame):
            self.data = self.params.rootfile
    
    def ReadExcludeFile(self):
        self.exclude = []
        if os.path.exists(os.path.join(self.params.data_path, 'exclude.txt')):
            with open(os.path.join(self.params.data_path, 'exclude.txt'), 'r') as file:
                self.exclude = [line.strip() for line in file.readlines()]
        elif isinstance(self.params.exclude, list):
            self.exclude = self.params.exclude


    def grabclasschoice(self,split):
        if split == 'test':
            if self.params.test_class_choice == 'same':
                class_choice = self.params.class_choice
                notcase = self.params.notcase
            else:
                class_choice = self.params.test_class_choice
                notcase = self.params.test_notcase
        else:
            class_choice = self.params.class_choice
            notcase = self.params.notcase
        return class_choice, notcase

    def FilterData(self):
        if len(self.exclude) != 0: 
            self.data = self.data[~self.data.index.isin(self.exclude)]
        #Check if the data is split
        if 'rxntype' not in self.data.columns:
            if not self._checkifsmilescolumnislist():
                datafilter = DenoteInputData(self.data,self.params)
                self.data['rxntype'] = self.data[self.params.smiles].apply(datafilter.getctype,args=(self.molecular,))
            else:
                for smicolumn in self.params.smiles:
                    datafilter = DenoteInputData(self.data,self.params)
                    self.data[f'rxntype_{smicolumn}'] = self.data[smicolumn].apply(datafilter.getctype,args=(self.molecular,))
        class_choice, notcase = self.grabclasschoice(self.split)
        datafilter.GrabData(class_choice=class_choice,notcase=notcase,split=self.split)
        self.data = datafilter.data   

    def _splitcreator(self):
        self.splitter = Splitter(self.params)
        if self.params.n_splits is None: self.data = self.splitter.onefold(self.data)
        else: self.data = self.splitter.xfold(self.data)

    def _obtaindesiredsplit(self):
        if self.split in ['train','val','test']:
            if self.params.fold is None:
                return self.data[self.data['split'] == self.split]
            else:
                return self.data[self.data[f'fold{self.params.fold}_split'] == self.split]
        elif self.split in ['trainval','traintest','valtest']:
            if self.params.fold is None:
                return self.data[self.data['split'].isin(self.split)]
            else:
                return self.data[self.data[f'fold{self.params.fold}_split'].isin(self.split)]

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
        if isinstance(columns,list):
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
   