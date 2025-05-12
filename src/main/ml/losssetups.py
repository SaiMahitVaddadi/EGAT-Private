import torch,logging
import torch.nn as nn
from ...loss.chemprop import MCCClassLoss,MCCMultiClassLoss,DirchletMultiClassLoss
from ...loss.spectra import SpectraWL,SpectraKLD,SpectraSIDLoss,SpectraTMSELoss,SpectraSISLoss
from ...loss.classification import ROCAUCLoss
from dataclasses import dataclass
from typing import Union, List

# Add the written loss functions to the dictionary

@dataclass
class Params:
    loss: Union[str, List[str]]
    metric: Union[str, List[str]]


class LossSetup:
    def __init__(self,arguments):
        self.params = arguments
        self.logger = logging.getLogger(__name__)

    def loadloss(self,lossfcn):
        try:
            loss = getattr(nn,f"{lossfcn}Loss")
            loss_fn = loss()
        except Exception as e:
            print(f"Error loading loss function {lossfcn}: {e}")
            try:
                loss_fn = globals().get(lossfcn)
                loss_fn = loss_fn()
            except:
                raise ValueError(f'Loss function not {lossfcn} supported')
        return loss_fn
    
    def LoadLoss(self):
        self.loss = self.loadloss(self.params.loss)
    
    def LoadMetric(self):
        self.metric = self.loadloss(self.params.metric)