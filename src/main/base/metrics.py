from .multitask import MultitaskLossAggregation
from torch import nn



class Metrics(MultitaskLossAggregation):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments

    def grabmetric(self,pred,target):
        return self.singulareval(pred, target, self.metric)
    
    def GetSingularMetric(self,pred,target):
        if 'multi' in self.params.model_type and self.params.metric_agg != None:
            aggloss = getattr(self,f'{self.params.metric_agg}MeanMetric')
            loss = aggloss(pred, target)
        elif 'multi' in self.params.model_type and self.params.tweights != None:
            loss = self.WeightedLossMetric(pred, target)
        else:
            loss = self.grabmetric(pred, target)
        return loss
    
    def WeightedLossMetric(self,pred,target):
        if len(self.params.tweights) == len(self.params.targets):
            loss = None
            for index,weight in enumerate(self.params.tweights):
                p = pred[:,index].unsqueeze(1)
                t = target[:,index].unsqueeze(1)
                if loss is not None: loss += weight * self.grabmetric(p,t)
                else: loss = weight * self.grabmetric(p,t)
        else:
            raise ValueError('Cannot work because the weights are underdetermined.')
        return loss 
    
    def setweight(self):
        if self.params.tweights == None:
            self.params.tweights = [1] * len(self.params.targets)
    
    def ArithMeanMetric(self, pred, target):
        loss = None
        self.setweight()
        for index,weight in enumerate(self.params.tweights):
            p = pred[:,index].unsqueeze(1)
            t = target[:,index].unsqueeze(1)
            if loss is not None: loss += weight * self.grabmetric(p,t)
            else: loss = weight * self.grabmetric(p,t)
        return loss / len(self.params.tweights)
    
    def GeomMeanMetric(self, pred, target):
        loss = None
        self.setweight()
        for index,weight in enumerate(self.params.tweights):
            p = pred[:,index].unsqueeze(1)
            t = target[:,index].unsqueeze(1)
            if loss is not None: loss *= weight * self.grabmetric(p,t)
            else: loss = weight * self.grabmetric(p,t)
        return loss / len(self.params.tweights)
    
    def HarmonicMeanMetric(self, pred, target):
        loss = None
        self.setweight()
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            current_loss = self.grabmetric(p, t)
            if loss is not None:
                loss += weight / current_loss
            else:
                loss = weight / current_loss
        return sum(self.params.tweights) / loss

    def QuadraticMeanMetric(self, pred, target):
        loss = None
        self.setweight()
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            current_loss = self.grabmetric(p, t)
            if loss is not None:
                loss += weight * (current_loss ** 2)
            else:
                loss = weight * (current_loss ** 2)
        return (loss / sum(self.params.tweights)) ** 0.5
    

    def grabsingularmetric(self,pred,target,m):
        if 'multi' in self.params.model_type and self.params.metric_agg != None:
            aggloss = getattr(self,f'{self.params.metric_agg}MeanMetricSingular')
            loss = aggloss(pred, target,m)
        elif 'multi' in self.params.model_type and self.params.tweights != None:
            loss = self.WeightedLossMetric(pred, target,m)
        else:
            loss = self.singulareval(pred, target,m)
        return loss
    
    def WeightedLossMetricSingular(self,pred,target,m):
        if len(self.params.tweights) == len(self.params.targets):
            loss = None
            for index,weight in enumerate(self.params.tweights):
                p = pred[:,index].unsqueeze(1)
                t = target[:,index].unsqueeze(1)
                if loss is not None: loss += weight * self.singulareval(p,t,m)
                else: loss = weight * self.singulareval(p,t,m)
        else:
            raise ValueError('Cannot work because the weights are underdetermined.')
        return loss 

    def ArithMeanMetricSingular(self, pred, target, m):
        loss = None
        self.setweight()
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            if loss is not None:
                loss += weight * self.singulareval(p, t, m)
            else:
                loss = weight * self.singulareval(p, t, m)
        return loss / len(self.params.tweights)

    def GeomMeanMetricSingular(self, pred, target, m):
        loss = None
        self.setweight()
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            if loss is not None:
                loss *= weight * self.singulareval(p, t, m)
            else:
                loss = weight * self.singulareval(p, t, m)
        return loss / len(self.params.tweights)

    def HarmonicMeanMetricSingular(self, pred, target, m):
        loss = None
        self.setweight()
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            current_loss = self.singulareval(p, t, m)
            if loss is not None:
                loss += weight / current_loss
            else:
                loss = weight / current_loss
        return sum(self.params.tweights) / loss

    def QuadraticMeanMetricSingular(self, pred, target, m):
        loss = None
        self.setweight()
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            current_loss = self.singulareval(p, t, m)
            if loss is not None:
                loss += weight * (current_loss ** 2)
            else:
                loss = weight * (current_loss ** 2)
        return (loss / sum(self.params.tweights)) ** 0.5

    def GetMultipleMetrics(self):
        losses = [] 
        for m in self.metric:
            losses.append(self.grabsingularmetric(self.pred,self.target,m))
        return losses
    
    def GetMetric(self):
        if isinstance(self.params.metric, str):
            loss = self.GetSingularMetric(self.pred,self.target)
        elif isinstance(self.params.metric, list):
            loss = self.GetMultipleMetrics()
        elif self.params.metric == None:
            loss = None
        else:
            raise ValueError(f"Metric {self.params.metric} not supported")
        return loss