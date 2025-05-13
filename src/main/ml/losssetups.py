import torch,logging
import torch.nn as nn
from ...loss.chemprop import MCCClassLoss,MCCMultiClassLoss,DirchletMultiClassLoss
from ...loss.spectra import SpectraWL,SpectraKLD,SpectraSIDLoss,SpectraTMSELoss,SpectraSISLoss
from ...loss.classification import ROCAUCLoss
from dataclasses import dataclass
from typing import Union, List

# Add the written loss functions to the dictionary

@dataclass
class LossParams:
    loss: Union[str, List[str]]
    metric: Union[str, List[str]]


class LossSetup:
    def __init__(self,arguments):
        self.params = arguments
        self.logger = logging.getLogger(__name__)

    def loadloss(self,lossfcn):
        if hasattr(nn, f"{lossfcn}Loss"):
            loss = getattr(nn,f"{lossfcn}Loss")
            loss_fn = loss()
        elif lossfcn in globals():
            loss = globals().get(lossfcn)
            loss_fn = loss()
        else:
            raise ValueError(f'Loss function not {lossfcn} supported')
        return loss_fn
    
    def LoadLoss(self):
        self.loss = self.loadloss(self.params.loss)

    def LoadSingularMetric(self):
        self.metric = self.loadloss(self.params.metric)
    
    def LoadMultipleMetrics(self):
        self.metric = []
        for m in self.params.metric:
            self.metric.append(self.loadloss(m))
    
    def LoadMetric(self):
        if isinstance(self.params.metric, str):
            self.LoadSingularMetric()
        elif isinstance(self.params.metric, list):
            self.LoadMultipleMetrics()
        else:
            raise ValueError(f"Metric {self.params.metric} not supported")