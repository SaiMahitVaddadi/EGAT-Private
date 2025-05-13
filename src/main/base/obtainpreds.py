from .scalers import ScalerSetup
from .csvsaver import CSVCommands
from .metrics import Metrics
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class PredsParams:
    Embed: bool = False
    AttnMaps: bool = False

#TO-ADD: BEP-Hr 

class ObtainPreds(CSVCommands,ScalerSetup,Metrics):
    def __init__(self, arguments):
        self.params = arguments

    def regularpreds(self,RGgs,PGgs,RAdd,PAdd):
        if self.params.graph == 'molecular':
            if self.params.addons is not None:
                if self.params.fingerprint == False:
                    pred = self.model(RGgs,RAdd)
                else:
                    pred = self.model(RAdd)
            else:
                pred = self.model(RGgs)
        elif self.params.graph == 'reaction':
            if self.params.addons is not None:
                if self.params.fingerprint == False:
                    pred = self.model(RGgs,PGgs,RAdd,PAdd)
                else:
                    pred = self.model(RAdd,PAdd)
            else:
                pred = self.model(RGgs,PGgs)
        return pred
    
    def additionalpreds(self,RGgs,PGgs,RAdd,PAdd,Hr):
        if self.params.graph == 'molecular':
            if self.params.addons is not None:
                if self.params.fingerprint == False:
                    pred = self.model(RGgs,RAdd,Hr)
                else:
                    pred = self.model(RAdd,Hr)
            else:
                pred = self.model(RGgs)
        elif self.params.graph == 'reaction':
            if self.params.addons is not None:
                if self.params.fingerprint == False:
                    pred = self.model(RGgs,PGgs,RAdd,PAdd,Hr)
                else:
                    pred = self.model(RAdd,PAdd,Hr)
            else:
                pred = self.model(RGgs,PGgs,Hr)
        return pred
    
    def bephrpreds(self,RGgs,PGgs,RAdd,PAdd,Hr2):
        if self.params.graph == 'molecular':
            if self.params.addons is not None:
                if self.params.fingerprint == False:
                    pred = self.model(RGgs,RAdd,Hr2)
                else:
                    pred = self.model(RAdd,Hr2)
            else:
                pred = self.model(RGgs)
        elif self.params.graph == 'reaction':
            if self.params.addons is not None:
                if self.params.fingerprint == False:
                    pred = self.model(RGgs,PGgs,RAdd,PAdd,Hr2)
                else:
                    pred = self.model(RAdd,PAdd,Hr2)
            else:
                pred = self.model(RGgs,PGgs,Hr2)
        return pred

    def bepfcn(self,pred,Hr): 
        pred = pred[:, 0].unsqueeze(1) * Hr + pred[:, 1].unsqueeze(1)
        return pred
    
    def checkifnonepredscase(self,pred,embeddings,Rmap,Pmap):
        pred = pred if 'RGgs' in locals() else None
        embeddings = embeddings if 'PGgs' in locals() else None
        Rmap = Rmap if 'RAdd' in locals() else None
        Pmap = Pmap if 'PAdd' in locals() else None   
        return pred,embeddings,Rmap,Pmap

    def grabpreds(self,RGgs,PGgs,RAdd,PAdd,Hr,Hr2=None):
        if self.params.model_type in ['direct','BEP','multi','BEP']:
            pred = self.regularpreds(RGgs,PGgs,RAdd,PAdd)
        elif self.params.model_type in ['Hr','Hr_multi','Hr2','Hr_multi2']:
            pred = self.additionalpreds(RGgs,PGgs,RAdd,PAdd,Hr)
        elif self.params.model_type in ['BEP_Hr']:
            pred = self.bephrpreds(RGgs,PGgs,RAdd,PAdd,Hr2)
        if self.params.model_type in ['BEP']: pred = self.bepfcn(pred,Hr)
        return pred

    def GetPrediction(self,RGgs,PGgs,RAdd,PAdd,Hr,Hr2=None):
        ###### GET PREDICTION
        pred = self.grabpreds(RGgs,PGgs,RAdd,PAdd,Hr,Hr2)
        if 'molecular' in self.params.graph:
            if not self.params.Embed:
                if self.params.AttnMaps:
                    pred,Rmap = pred
                else:
                    pred = pred
            else:
                if self.params.AttnMaps:
                    pred,embeddings,Rmap = pred
                else:
                    pred,embeddings = pred
        elif 'reaction' in self.params.graph:
            if not self.params.Embed:
                if self.params.AttnMaps:
                    pred,Rmap,Pmap = pred
                else:
                    pred = pred
            else:
                if self.params.AttnMaps:
                    pred,embeddings,Rmap,Pmap = pred
                else:
                    pred,embeddings = pred
        return pred,embeddings,Rmap,Pmap
    
    def createdataresults(self,data,pred,target,smiles,id,rtypes,embeddingsdata,embeddings):
        batch_result = torch.cat([pred,target]).cpu().data
        batch_result = torch.cat([smiles,batch_result]).numpy()
        batch_result = np.hstack((id,rtypes,batch_result)).tolist()
        if self.params.Embed: embeddingsdata.append(embeddings.cpu().numpy().tolist())
        data.append(batch_result)
        return data

    def AddtoDataset(self,data,pred,target,smiles,id,rtypes,embeddingsdata,embeddings):
        # Merge the torch Tensors, get them out of cuda, and move them to numpy 
        pred = self.InvertTarget(pred)
        target = self.InvertTarget(target)
        data = self.createdataresults(data,pred,target,smiles,id,rtypes,embeddingsdata,embeddings)
        loss = self.GetLoss(pred,target)
        metric = self.GetMetric(pred,target)
        return data,embeddingsdata,loss,metric