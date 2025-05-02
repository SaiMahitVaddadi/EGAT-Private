import torch,logging,os,hydra
from torch import nn
from dataclasses import dataclass
from typing import Optional

from .gpusetups import GPUSetup
from .losssetups import LossSetup
from .egatsetup import EGATModelSetup
from .optimizersetup import OptimizerSetup
from .schedulersetup import ScheduleSetup
from .utils import bn_momentum_adjust


class MLSetup(GPUSetup,LossSetup,EGATModelSetup,OptimizerSetup,ScheduleSetup):
    def __init__(self,arguments):
        super().__init__(arguments)
        ### Set Logger
        self.Load()

    def Load(self):
        self.LoadWandB()
        self.LoadTorchSetup()
        self.LoadLoss()
        self.LoadMetric()
        self.best_loss,self.start_epoch,self.checkpoint = self.LoadModel()
        self.model,self.total_params,self.trainable_params = self.ObtainModelParams()
        self.optimizer = self.LoadOptimizer(self.model)
        self.scheduler = self.LoadScheduler(self.optimizer)
        self.metric = self.LoadMetric()
        self.global_epoch = 0
        self.loss_increase_count = 0

    def LoadWandB(self):
        ### Check if weights and biases is needed for live model monitoring
        if self.params.weightsandbiases:
            import wandb
            wandb.init(project=self.params.wandbproject,name=self.params.wandbname)
    
    
    