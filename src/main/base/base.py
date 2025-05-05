import torch,os
from torch import nn
from ..ml.setup import MLSetup
from ...loader.loader import EGATDataLoader
import numpy as np 
from tqdm import tqdm
import pandas as pd 
from dataclasses import dataclass, field
from typing import List, Optional, Union


class TensorCommands:
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments
    




class InputSetups(MLSetup):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments
    
    def LoadData(self):
        self.loader = EGATDataLoader(self.params)
    
    def LoadScalers(self):
        try:
            self.scaler = self.loader.egatdataset['train'].target_normalizer
        except:
            self.scaler = None 

        try:
            self.scaler = self.loader.egatdataset['train'].additional_normalizer
        except:
            self.scaler = None 

    def LoadLearningRate(self,epoch):
        if epoch < self.params.epoch:
            lr = self.get_learning_rate(self.scheduler)
            return lr
        else:
            return self.params.learning_rate_min
    
    def UpdateLR(self,epoch):
        if self.params.scheduler == 'plateau':
            self.scheduler.step(self.best_loss)
        else:
            self.scheduler.step()
        self.params.learning_rate = self.LoadLearningRate(epoch)
        self.logger.info(f"Learning rate is set to {self.params.learning_rate}.")
    
    def UpdateMomentum(self,epoch):
        momentum = self.MOMENTUM_ORIGINAL * (self.MOMENTUM_DECAY ** (epoch //self.MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        return momentum
    
    def GrabTargets(self,targets):
        if self.params.model_type in ['direct','BEP','Hr']:
            target    = torch.Tensor([float(i[2]) for i in targets]).view(self.params.batch_size,1).to(self.device)
        elif self.params.model_type in ['multi','Hr_multi']:
            target = targets.float().view(self.params.batch_size,len(self.params.target)).to(self.device)
        
        self.mask = ~torch.isnan(target)
        return target 
    
    def GetHr(self,additionals):
        if self.params.Norm is not None:
            Hr = self.ScaleData(self.scaler,additionals)
        else:
            if isinstance(self.params.additionals,list):
                Hr = additionals.float().view(self.params.batch_size,len(self.params.additionals)).to(self.device)
            else:
                Hr = torch.Tensor([float(i[0]) for i in additionals]).view(self.params.batch_size,1).to(self.device)
        return Hr


    def GrabAdditionals(self,additionals):
        if self.params.model_type == 'BEP':    
            if isinstance(self.params.additionals,list):
                raise ValueError('Error: BEP-like Prediction can only be done on one additional set of values.')
            else:
                Hr = torch.Tensor([float(i[0]) for i in additionals]).view(self.params.batch_size,1).to(self.device)
        elif self.params.model_type == 'Hr' or self.params.model_type == 'Hr_multi':
            Hr = self.GetHr(additionals)
        
        return Hr
    
        