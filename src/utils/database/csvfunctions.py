import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

'''
TO-DO:
- Add support for dask, pyspark, sql, and polars
- Add functions for mongodb and postgresql
'''


@dataclass
class DenoteInputDataParams:
    denoteby: Optional[str] = None
    split: Optional[str] = None
    smilescol: Optional[str] = None
    fold: Optional[int] = None

class DenoteInputData:
    def __init__(self, data: pd.DataFrame, params: DenoteInputDataParams):
        """
        Initialize the DenoteInputData class. 
        
        For reaction studies, this labels and divides the data by: 

        1) Molecularity
        2) Reaction Type (Break 2, form 2 and what not)
        3) Size of the Atoms

        For reaction studies, this labels and divides the data by:
        
        1) # of Components
        2) Size of the Atoms
        3) # of Rings
        4) # of Aromatic Rings
        
        Parameters:
        --------------------------------------------------------------------
        data (pd.DataFrame): The input data.
        params (DenoteInputDataParams): The parameters for denotation and split.
        """
        self.data = data 
        self.params = params

    def GetSplit(self,split=None):
        """
        Filter the data based on the split type.
        """
        if self.params.fold is not None:
            coltolook = f'fold{self.params.fold}_split'
        else:
            coltolook = 'split'
        
        if split in ['train', 'test', 'val']:
            self.data = self.data[self.data[coltolook] == split]
        elif split == 'traintest':
            self.data = self.data[self.data[coltolook].isin(['train', 'test'])]
        elif split == 'trainval':
            self.data = self.data[self.data[coltolook].isin(['train', 'val'])]
        elif split == 'testval':
            self.data = self.data[self.data[coltolook].isin(['test', 'val'])]
        else:
            self.data = self.data
    
    def _obtaindenotationstrcase(self,class_choice,notcase=False,coltolook = 'rxntype'):
        if notcase:
            self.data = self.data[~self.data[coltolook].str.contains(class_choice, case=False)]
        else:
            self.data = self.data[self.data[coltolook].str.contains(class_choice, case=False)]
    
    def _obtaindenotationlist(self,class_choice,notcase=False,coltolook = 'rxntype'):
        if notcase:
            self.data = self.data[~self.data[coltolook].isin(class_choice)]
        else:
            self.data = self.data[self.data[coltolook].isin(class_choice)]

    def _obtaindenotationdict(self,class_choice,notcase=False):
        for key, value in class_choice.items():
            if isinstance(value, str):
                if isinstance(notcase,dict):
                    self._obtaindenotationstrcase(value, notcase.get(key, False),coltolook=f'rxntype_{key}')
                else:
                    self._obtaindenotationstrcase(value, notcase,coltolook=f'rxntype_{key}')
            elif isinstance(value, list):
                if isinstance(notcase,dict):
                    self._obtaindenotationlist(value, notcase.get(key, False),coltolook=f'rxntype_{key}')
                else:
                    self._obtaindenotationlist(value, notcase,coltolook=f'rxntype_{key}')

    def GetDenotation(self,class_choice=None,notcase=False):
        """
        Filter the data based on the denotation type.
        """
        if class_choice == None:
            self.data = self.data
        if isinstance(class_choice, str):
            self._obtaindenotationstrcase(class_choice, notcase)
        elif isinstance(class_choice, list):
            self._obtaindenotationlist(class_choice, notcase)
        elif isinstance(class_choice, dict):
            self._obtaindenotationdict(class_choice, notcase)
        else:
            self.data = self.data
    
    @staticmethod
    def getctype(smi: str, molecular: bool = False) -> str:
        """
        Get the reaction type based on the SMILES string.

        Parameters:
        smi (str): The SMILES string.
        molecular (bool, optional): Whether to consider molecular reactions.

        Returns:
        str: The reaction type.
        """
        if molecular:
            return f"R{len(smi.split('.'))}"
        else:
            smi = smi.split('>>')
            return f"R{len(smi[0].split('.'))}P{len(smi[1].split('.'))}"

    def GrabData(self,split=None,class_choice=None,notcase=False):
        self.GetDenotation(class_choice,notcase)
        self.GetSplit(split)


