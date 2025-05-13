from .base.base import MLTrainandPredictBase
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

@dataclass
class Params:
    epoch: int
    epoch_const: int
    learning_rate: float
    lr_decay: float
    step_size: int
    expdecay: float
    warmup: int
    scheduler: str
    batch_size: int
    num_workers: int
    randomize: bool
    norm: Optional[str] = None
    scale_region: Optional[str] = None
    test_only: bool = False
    molecular: bool = False
    target: Union[str, List[str]] = 'target'
    tweights: List[float] = field(default_factory=list)
    model_type: str = 'direct'
    additionals: Optional[Union[str, List[str]]] = None
    hasaddons: bool = False
    Embed: bool = False
    AttnMaps: bool = False
    normtarget: bool = False
    trainedonnorm: bool = False
    metric: Optional[str] = None
    loss: Union[str, List[str]] = 'MAE'
    loss_agg: str = 'Arith'
    loss_weights: Optional[List[float]] = None
    patience: int = 10
    loss_threshold: float = 1e-4
    weightsandbiases: bool = False
    save_style: str = 'best'

class Train(MLTrainandPredictBase):
    def __init__(self, arguments):
        self.params = arguments


    def startepoch(self):
        self.start_epoch = 0
        
    def basetrainfcn(self,epoch,ensemble,fold):
        self.loss_increase_count += 1
        self.logger.info('Epoch %d (%d/%s):' % (self.global_epoch + 1, epoch + 1, self.params.epoch)) 
        validation_loss = self.TrainIteration(epoch,ensemble=ensemble,fold=fold)
        self.logger.info('Validation loss: %.5f' % validation_loss)
        return validation_loss
        
    
    def TrainbyEpoch(self,ensemble=None,fold=None):
        self.startepoch()
        self.LoadMLNecessities()
        for epoch in range(self.start_epoch, self.params.epoch+self.params.epoch_const):
            self.basetrainfcn(epoch,ensemble,fold)
            if self.checkpatience():
                self.logger.info('Early stopping triggered.')
                break
            else:
                self.global_epoch += 1
        
            
    
    def TrainUntilConvergence(self,ensemble=None,fold=None):
        prev_loss = float('inf')
        self.startepoch()
        epoch = self.start_epoch
        self.LoadMLNecessities()
        while abs(prev_loss - validation_loss) >= self.params.loss_threshold:
            validation_loss = self.basetrainfcn(epoch,ensemble,fold)

            if epoch == self.params.epoch + self.params.epoch_const: break
            if self.checkpatience(): break
            else:
                self.global_epoch += 1
                epoch += 1
            prev_loss = validation_loss
            
    def Train(self,ensemble=None,fold=None):
        if self.params.loss_threshold is not None:
            loss = self.TrainUntilConvergence(ensemble,fold)
        else:
            loss = self.TrainbyEpoch(ensemble,fold)
        return loss 

    def TrainEnsemble(self,fold=None):
        losses = []
        for model_idx in range(self.params.ensemble):
            self.logger.info(f'Training model {model_idx + 1}/{self.params.ensemble}')
            loss = self.Train(model_idx,fold=fold) # Train the model
            losses.append(loss)
        return np.mean(losses)
            
    def TrainCV(self):
        losses = []
        for fold in range(self.params.fold):
            self.logger.info(f'Training fold {fold + 1}/{self.params.fold}')
            self.params.fold = fold
            if self.params.ensemble > 1:
                self.logger.info(f'Training ensemble for fold {fold + 1}/{self.params.fold}')
                loss = self.TrainEnsemble(fold)
            else:
                loss = self.Train(fold=fold)
            
            losses.append(loss)
        return np.mean(losses)

    def TrainingProtocol(self):
        if self.params.ensemble > 1 and not self.params.crossval:
            loss = self.TrainEnsemble()    
        elif self.params.crossval:
            loss = self.TrainCV()
        else:
            loss = self.Train()
        return loss
    

    
    

    