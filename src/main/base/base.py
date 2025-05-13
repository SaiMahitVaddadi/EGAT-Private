from .loops import Loops
import wandb
import numpy as np
from .torchsaver import TorchSaver
from dataclasses import dataclass

@dataclass
class BaseEGATTrainParams:
    weightsandbiases: bool = False
    test_only: bool = False
    patience: int = 10
    


class MLTrainandPredictBase(Loops,TorchSaver):

    def __init__(self, arguments):
        self.params = arguments   
    
    def LoadMLNecessities(self):
        self.LoadMLSetup()
        self.LoadScalers()
        
    def UpdateWandB(self,train,test,val,epoch,lr):
        # compute the average
        if self.params.weightsandbiases:
            if self.params.test_only:
                wandb.log({"learning_rate":lr, "train_loss": np.mean(train), 'val_loss': np.mean(val)},step=epoch)
            else:
                wandb.log({"learning_rate":lr, "train_loss": np.mean(train), 'val_loss': np.mean(val),'ext_loss': np.mean(test)},step=epoch)

    def checkpatience(self):
        self.global_epoch += 1
        if self.loss_increase_count > self.params.patience: return True
        else: return False
    
    def TrainIteration(self,epoch,ensemble=None,fold=None):
        train,val,test,train_embeddings,val_embeddings,test_embeddings,train_loss_list,val_loss_list,test_loss_list,lr,train_metrics_list,val_metrics_list,test_metrics_list = self.TrainingLoop(epoch)
        self.UpdateWandB(train,val,test,train_embeddings,val_embeddings,test_embeddings,train_loss_list,val_loss_list,test_loss_list,lr)
        self.SaveTorchModel(train,val,test,train_embeddings,val_embeddings,test_embeddings,val_loss_list,train_loss_list,epoch,self.columns,ensemble,fold)
        self.SaveLossesToCSV(train_loss_list, val_loss_list, test_loss_list, lr, epoch,train_metrics_list,val_metrics_list,test_metrics_list,ensemble,fold)
        return np.mean(val_loss_list)

    def PredictRun(self):
        self.LoadMLNecessities()
        val,val_embeddings,val_loss_list,val_metrics_list = self.PredictLoop()
        self.SavePredictedData(val)
        self.SavePredictedEmbeddings(val_embeddings)
        # Log val_loss_list and val_metrics using np.mean and logger
        if hasattr(self, 'logger'):
            self.logger.info(f"Validation Loss: {np.mean(val_loss_list)}")
            self.logger.info(f"Validation Metrics: {np.mean(val_metrics_list)}")

