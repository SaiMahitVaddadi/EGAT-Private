import torch,os
from torch import nn
from ..ml.setup import MLSetup
from ...loader.loader import EGATDataLoader
import numpy as np 
from tqdm import tqdm
import pandas as pd 
from dataclasses import dataclass, field
from typing import List, Optional, Union

class ScalerSetup(MLSetup):

    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments
        self.Load()    
    
    def LoadScalers(self):
        if hasattr(self.loader.egatdataset['train'], 'target_normalizer'):
            self.target_scaler = self.loader.egatdataset['train'].target_normalizer
        else:
            self.target_scaler = None 

        if hasattr(self.loader.egatdataset['train'], 'additional_normalizer'):
            self.additional_scaler = self.loader.egatdataset['train'].additional_normalizer
        else:
            self.additional_scaler = None 

    def baseinversefcn(self,scaler,data):
        return scaler.inverse_transform(data)
    
    def InvertTarget(self,data):
        if self.target_scaler is not None:
            data = self.target_scaler.inverse_transform(data)
        return data
    
    def InvertAdditional(self,data):
        if self.additional_scaler is not None:
            data = self.additional_scaler.inverse_transform(data)
        return data
    