import argparse
import os
import torch
import logging
import sys
import importlib
import shutil
import numpy as np
from tqdm import tqdm
import hydra
import omegaconf
import pandas as pd 
from torch.autograd import Variable
import joblib

from Dataset import EGATDataset
from collate.molecule import MolecularCollator
from collate.reaction import ReactionCollator
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataLoaderParams:
    data_path: str
    exclude: Optional[str] = None
    test_only: bool = False
    root: str = ''
    class_choice: Optional[str] = None
    randomize: bool = False
    fold: Optional[int] = None
    foldtype: Optional[str] = None
    size: Optional[int] = None
    target: Optional[str] = None
    additionals: Optional[List[str]] = None
    addons: bool = False
    molecular: bool = False
    batch_size: int = 32

class EGATDataLoader:
    def __init__(self,arguments):
        self.params = arguments

    def Exclude(self):
        ### Get the files to exclude. If not, just set it blank. 
        if self.params.exclude is not None:
            if os.path.isfile(os.path.join(self.params.data_path,self.params.exclude)): # Check if path exists
                exclude = []
                with open(os.path.join(self.params.data_path,self.params.exclude),'r') as f: # open file 
                    for lc,lines in enumerate(f):
                        exclude.append(lines.split('/')[-1].split('.json')[0]) # Add exclude files 
            else:
                self.logger.info(f'{os.path.join(self.params.data_path,self.params.exclude)} does not exist.')
                exclude = []
        else:
            self.logger.info(f'{os.path.join(self.params.data_path)} has not exclude file.')
            exclude = []
        
        return exclude 
    

    

    def LoadDataset(self):
        self.data = dict()
        if self.params.test_only:
            splits = ['train','test']
        else:
            splits = ['train','val','test']

        exclude = self.Exclude()
        for split in splits:
            self.data[split] = EGATDataset(root=self.params.root,split=split, class_choice=self.params.class_choice, exclude=exclude,randomize=self.params.randomize,fold=self.params.fold,foldtype=self.params.foldtype,size=self.params.size,target=self.params.target,additional=self.params.additionals,hasaddons=self.params.addons,molecular=self.params.graph)

    
    def GrabCollateFunction(self):
        if self.params.hasaddons: #Check if we need RDKit Global Features. If we do, load them.
            if self.params.additionals is not None: # Check if there are added features. If we do, load them.
                if self.params.graph == 'molecule': # Check if we only need molecular features. If we do, only load R features. 
                    self.collator = MolecularCollator.allprops
                elif self.params.graph == 'reaction':
                    self.collator = ReactionCollator.allprops
            else:
                if self.params.graph == 'molecule': # Check if we only need molecular features. If we do, only load R features. 
                    self.collator = MolecularCollator.addons
                elif self.params.graph == 'reaction':
                    self.collator = ReactionCollator.addons
        else:
            if self.params.additionals is not None:
                if self.params.graph == 'molecule':
                    self.collator = MolecularCollator.additionals
                elif self.params.graph == 'reaction':
                    self.collator = ReactionCollator.additionals
            else:
                if self.params.graph == 'molecule':
                    self.collator = MolecularCollator.targets
                elif self.params.graph == 'reaction':
                    self.collator = ReactionCollator.targets
            

    def LoadData(self):
        self.LoadDataset()
        self.GrabCollateFunction()
        self.egatloader = dict()
        if self.params.test_only:
            splits = ['train','test']
        else:
            splits = ['train','val','test']
        for split in splits:
            self.egatloader[split] = torch.utils.data.DataLoader(self.data[split], batch_size=self.params.batch_size, shuffle=True, collate_fn=self.collator)
    
    def __call__(self):
        self.LoadData()
