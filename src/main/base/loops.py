from .batchiteration import BatchIteration
from .learningrates import LearningRateFunctions
from .torchsaver import TorchSaver
import numpy as np
import torch
from ..ml.utils import bn_momentum_adjust

class Loops(BatchIteration,LearningRateFunctions):
    def __init__(self, arguments):
        self.params = arguments

    def lrfunc(self,epoch):
        lr = self.LoadLearningRate(epoch)
        self.UpdateLR(lr)
        momentum = self.UpdateMomentum(epoch)
        self.logger.info('BN momentum updated to: %f' % momentum)
        self.model = self.model.apply(lambda x: bn_momentum_adjust(x, momentum))
        self.model = self.model.train()
        traindf,valdf,testdf = self.CreateAllCSVs()
        return lr,traindf,valdf,testdf

    def copyforaugtrain(self,train):
        augtrain = train.copy()
        return augtrain

    def trainiteration(self,epoch,train,sets='train'):
        self.logger.info('Training...')
        train,train_embeddings,train_loss_list,train_metrics_list = self.Iterate(self.data[sets],mode = sets)
        return train,train_embeddings,train_loss_list,train_metrics_list

    def setblockstoeval(self):
        self.model = self.model.eval()

    def evaliteration(self,epoch,val,test):
        self.logger.info('Checking...')
        with torch.no_grad():
            self.setblockstoeval()
            val,val_embeddings,val_loss_list,val_metrics_list = self.Iterate(self.data['val'],mode = 'val')
            if not self.params.test_only: 
                test,test_embeddings,test_loss_list,test_metrics_list = self.Iterate(self.data['test'],mode = 'test')
            else:
                test = None
                test_embeddings = None
                test_loss_list = None
                test_metrics_list = None
        return val,val_embeddings,test,test_embeddings,val_loss_list,test_loss_list,val_metrics_list,test_metrics_list
    

    def prediteration(self,val,epoch=None,sets='all'):
        self.logger.info('Checking...')
        with torch.no_grad():
            self.setblockstoeval()
            val,val_embeddings,val_loss_list,val_metrics_list = self.Iterate(self.data[sets],mode = sets)
        return val,val_embeddings,val_loss_list,val_metrics_list

    def TrainingLoop(self,epoch):
        '''Adjust learning rate and BN momentum'''
        # set up lr
        lr,train,val,test = self.lrfunc(epoch)
        train,train_embeddings,train_loss_list,train_metrics_list = self.trainiteration(epoch,train,sets='train')
        # update momentum
        val,val_embeddings,test,test_embeddings,val_loss_list,test_loss_list,val_metrics_list,test_metrics_list = self.evaliteration(epoch,val,test)
        return train,val,test,train_embeddings,val_embeddings,test_embeddings,train_loss_list,val_loss_list,test_loss_list,lr,train_metrics_list,val_metrics_list,test_metrics_list
    
    def PredictLoop(self):
        test = self.CreateCSV()
        self.logger.info('Predicting...')
        val,val_embeddings,val_loss_list,val_metrics_list = self.prediteration(val=test,sets='all')
        return val,val_embeddings,val_loss_list,val_metrics_list


    