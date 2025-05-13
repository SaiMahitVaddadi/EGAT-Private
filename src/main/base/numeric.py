from ..ml.setup import MLSetup
import torch

class NumericTensorFunctions(MLSetup):

    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments 

    def BaseGrabFcn(self,vector,mask=True):
        stackedvector = torch.stack(vector).float().to(self.device)
        if mask ==True: self.mask = ~torch.isnan(stackedvector)
        return stackedvector
    
    def GrabTargets(self,targets):
        target = self.BaseGrabFcn(targets,mask=True)
        return target 
    
    def GrabAdditionals(self,additionals):
        return self.BaseGrabFcn(additionals,mask=False)
        
