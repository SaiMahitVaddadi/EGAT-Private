import torch,logging
import torch.nn as nn
from ...loss.chemprop import MCCClassLoss,MCCMultiClassLoss,DirchletMultiClassLoss
from ...loss.spectra import SpectraWL,SpectraKLD,SpectraSIDLoss,SpectraTMSELoss,SpectraSISLoss
from ...loss.classification import ROCAUCLoss

# Add the written loss functions to the dictionary

class LossSetup:
    def __init__(self,arguments):
        self.params = arguments
        self.logger = logging.getLogger(__name__)

    def loadloss(self,lossfcn):
        try:
            loss = getattr(torch,nn,f"{lossfcn}Loss")
            loss_fn = loss()
        except:
            loss_fn = None
            raise ValueError(f'Loss function not {lossfcn} supported')
        return loss_fn
    
    def baseloader(self,losses):
        ### Load the loss function
        if isinstance(losses,str):
            loss_fn = self.LossLoader(losses)
        elif isinstance(losses,list):
            loss_fn = []
            for loss in losses:
                loss_fn.append(self.LossLoader(loss))
        return loss_fn
    
    def LoadLoss(self):
        self.loss = self.loadloss(self.params.loss)
    
    def LoadMetric(self):
        self.metric = self.loadloss(self.params.metric)