from ..ml.setup import MLSetup
from torch import nn
class MultitaskLossAggregation(MLSetup):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments

    def singulareval(self, pred, target,lossfcn):
        if self.mask != None: loss = lossfcn(pred[self.mask], target[self.mask])
        else: loss = lossfcn(pred, target)
        return loss
    
    def grabloss(self,pred,target):
        return self.singulareval(pred, target, self.loss)
    
    def GetLoss(self,pred,target):
        if 'multi' in self.params.model_type and self.params.loss_agg != None:
            aggloss = getattr(self,f'{self.params.loss_agg}Mean')
            loss = aggloss(pred, target)
        elif 'multi' in self.params.model_type and self.params.tweights != None:
            loss = self.WeightedLoss(pred, target)
        else:
            loss = self.grabloss(pred, target)
        return loss

    def WeightedLoss(self,pred,target):
        if len(self.params.tweights) == len(self.params.targets):
            loss = None
            for index,weight in enumerate(self.params.tweights):
                p = pred[:,index].unsqueeze(1)
                t = target[:,index].unsqueeze(1)
                if loss is not None: loss += weight * self.grabloss(p,t)
                else: loss = weight * self.grabloss(p,t)
        else:
            raise ValueError('Cannot work because the weights are underdetermined.')
        return loss 

    def setweight(self):
        if self.params.tweights == None:
            self.params.tweights = [1] * len(self.params.targets)
    
    def ArithMean(self, pred, target):
        loss = None
        self.setweight()
        for index,weight in enumerate(self.params.tweights):
            p = pred[:,index].unsqueeze(1)
            t = target[:,index].unsqueeze(1)
            if loss is not None: loss += weight * self.grabloss(p,t)
            else: loss = weight * self.grabloss(p,t)
        return loss / len(self.params.tweights)
    
    def GeomMean(self, pred, target):
        loss = None
        self.setweight()
        for index,weight in enumerate(self.params.tweights):
            p = pred[:,index].unsqueeze(1)
            t = target[:,index].unsqueeze(1)
            if loss is not None: loss *= weight * self.grabloss(p,t)
            else: loss = weight * self.grabloss(p,t)
        return loss / len(self.params.tweights)
    
    def HarmonicMean(self, pred, target):
        loss = None
        self.setweight()
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            current_loss = self.grabloss(p, t)
            if loss is not None:
                loss += weight / current_loss
            else:
                loss = weight / current_loss
        return sum(self.params.tweights) / loss

    def QuadraticMean(self, pred, target):
        loss = None
        self.setweight()
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            current_loss = self.grabloss(p, t)
            if loss is not None:
                loss += weight * (current_loss ** 2)
            else:
                loss = weight * (current_loss ** 2)
        return (loss / sum(self.params.tweights)) ** 0.5
    

    