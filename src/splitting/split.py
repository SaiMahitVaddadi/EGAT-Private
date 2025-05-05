from sqlite3 import SQLITE_ALTER_TABLE
import sqlite3
import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import omegaconf
import hydra
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import numpy as np
from astartes.molecules import train_val_test_split_molecules
from copy import deepcopy
import pandas as pd 
from tqdm import tqdm
from alive_progress import alive_bar
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AstartesParams:
    hopts: dict
    fingerprint: str
    fingerprint_args: dict

@dataclass
class SplitParams:
    split_type: str
    train_size: float
    test_size: float
    random_state: int
    n_splits: int
    fold_shuffle: bool
    smiles: str
    target: str
    astartes: AstartesParams


class Splitter:
    def __init__(self,arguments):
        self.params = arguments 


    def onefold(self,df):
        if self.params.split_type == 'random':
            train_df, temp_df = train_test_split(df.index.tolist(), test_size=1-self.params.train_size, random_state=self.params.random_state)
            val_df, test_df = train_test_split(temp_df, test_size=.5, random_state=self.params.random_state)
            return train_df, val_df, test_df
        else:
            
            train,val,test = train_val_test_split_molecules(molecules=df[self.params.smiles].values,y=df.loc[self.params.target].values,train_size=self.params.train_size,val_size = (1-self.params.train_size)/2, test_size = (1-self.params.train_size)/2,
                                                        sampler=self.params.split_type,hopts=self.params.astartes.hopts,fingerprint=self.params.astartes.fingerprint,fprints_hopts=self.params.astartes.fingerprint_args)
            return train,val,test
                                                 
                                                    
    def xfold(self,df):
        # Create a withheld test set first
        train_val_df, test_df = train_test_split(df, test_size=self.params.test_size, random_state=self.params.random_state)
        
        # Perform k-fold on the remaining data
        kf = KFold(n_splits=self.params.n_splits, shuffle=self.params.fold_shuffle, random_state=self.params.random_state)
        splits = []
        for train_index, val_index in kf.split(train_val_df):
            train_df = train_val_df.iloc[train_index]
            val_df = train_val_df.iloc[val_index]
            splits.append((train_df, val_df, test_df))
        return splits
