import pandas as pd

'''
TO-DO:
- Add support for dask, pyspark, sql, and polars
- Add functions for mongodb and postgresql
'''


class DenoteInputData:
    def __init__(self, data, denoteby: str=None, split: str=None, smilescol: str=None,fold:int = None):
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
        denoteby (str, optional): The column to denote by.
        split (str, optional): The split type (e.g., 'train', 'test', 'val').
        smilescol (str, optional): The column containing SMILES strings.
        """
        self.data = data 
        self.denoteby = denoteby
        self.split = split

    def GetSplit(self):
        """
        Filter the data based on the split type.
        """
        if self.fold is not None: self.data = self.data[self.data.fold == self.fold]
        if self.split in ['train', 'test', 'val']:
            self.data = self.data[self.data.split == self.split]
        elif self.split == 'traintest':
            self.data = self.data[self.data.split.isin(['train', 'test'])]
        elif self.split == 'trainval':
            self.data = self.data[self.data.split.isin(['train', 'val'])]
        elif self.split == 'testval':
            self.data = self.data[self.data.split.isin(['test', 'val'])]
        else:
            self.data = self.data
    
    def GetDenotation(self):
        """
        Filter the data based on the denotation type.
        """
        if isinstance(self.denoteby, list):
            self.data = self.data[self.data.rxntype.isin(self.denoteby)]
        elif isinstance(self.denoteby, str):
            self.data = self.data[self.data.rxntype == self.denoteby]
        else:
            self.data = self.data
    
    def getctype(smi, molecular=False):
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


