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

    def GetSplit(self):
        """
        Filter the data based on the split type.
        """
        if self.params.fold is not None:
            self.data = self.data[self.data.fold == self.params.fold]
        if self.params.split in ['train', 'test', 'val']:
            self.data = self.data[self.data.split == self.params.split]
        elif self.params.split == 'traintest':
            self.data = self.data[self.data.split.isin(['train', 'test'])]
        elif self.params.split == 'trainval':
            self.data = self.data[self.data.split.isin(['train', 'val'])]
        elif self.params.split == 'testval':
            self.data = self.data[self.data.split.isin(['test', 'val'])]
        else:
            self.data = self.data
    
    def GetDenotation(self):
        """
        Filter the data based on the denotation type.
        """
        if isinstance(self.params.denoteby, list):
            self.data = self.data[self.data.rxntype.isin(self.params.denoteby)]
        elif isinstance(self.params.denoteby, str):
            self.data = self.data[self.data.rxntype == self.params.denoteby]
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

    def GrabData(self):
        self.GetDenotation()
        self.GetSplit()


