from ..ml.setup import MLSetup
from .csvsaver import CSVCommands
import numpy as np
import torch

class TorchSaver(CSVCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments
        

    def setupinitals(self,val_loss_list):
        self.loss_increase_count = 0
        self.best_loss = np.mean(val_loss_list)
        self.logger.info('Save model...')
    
    def addextratopath(self,ensemble=None, fold=None):
        if ensemble is not None: modelname += f'_ensemble_{ensemble}'
        if fold is not None: modelname += f'_fold_{fold}'
        modelname += '.pth'
        self.logger.info('Saving at %s' % modelname)
        return modelname
    
    def bestmodelfilepath(self,ensemble=None, fold=None):
        modelname = 'best_model'
        modelname = self.addextratopath(ensemble, fold)
        return modelname

    def grabstate(self,epoch,train_loss_list):
        state = {
                'epoch': epoch,
                'train_acc': np.mean(train_loss_list),
                'test_acc': self.best_loss,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
        return state

    def saveegatmodel(self,train_loss_list,val_loss_list,epoch=None,ensemble=None,fold=None):
        self.setupinitals(val_loss_list)
        savepath = self.bestmodelfilepath(ensemble, fold)
        state = self.grabstate(epoch,train_loss_list)
        
        torch.save(state, savepath)
        self.logger.info('Saving model....')

    def checkinterval(self,epoch):
        interval = int(self.params.save_style.split('-')[1])
        return epoch % interval == 0

    def epochmodelfilepath(self,epoch=None,ensemble=None, fold=None):
        modelname = f'epoch_{epoch}'
        modelname = self.addextratopath(ensemble, fold)
        return modelname

    def SaveTorchModel(self,train,val,test,train_embeddings,val_embeddings,test_embeddings,val_loss_list,train_loss_list,epoch,columns,ensemble=None,fold=None):
        if np.mean(val_loss_list) < self.best_loss:  
            self.saveegatmodel(train_loss_list,val_loss_list,epoch,ensemble,fold)
            if self.params.save_style == 'best':
                self.SaveData(train,val,test,columns,ensemble,fold)
                self.SaveEmbeddingsData(train_embeddings,val_embeddings,test_embeddings,ensemble,fold)
        else:
            if 'every-' in self.params.save_style and self.checkinterval(epoch):    
                self.logger.info('Save model...')
                savepath = self.epochmodelfilepath(epoch,ensemble, fold)
                state = self.grabstate(epoch,train_loss_list)
                torch.save(state, savepath)
                self.logger.info('Saving model....')
                self.SaveData(train,val,test,columns,ensemble,fold,epoch)
                self.SaveEmbeddingsData(train_embeddings,val_embeddings,test_embeddings,ensemble,fold,epoch)
    
        
