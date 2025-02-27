from tqdm import tqdm
import torch
import numpy as np 
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,QuantileTransformer,PowerTransformer,FunctionTransformer,KBinsDiscretizer
import omegaconf
import joblib
from dataclasses import dataclass, field
from typing import List, Optional, Union, Type
from sklearn.base import BaseEstimator

@dataclass
class NormalizerParams:
    batch_size: int
    additionals: Optional[Union[List[str], omegaconf.listconfig.ListConfig]] = None
    targets: Optional[Union[List[str], omegaconf.listconfig.ListConfig]] = None
    hasaddons: bool = False
    molecular: bool = False
    normtarget: bool = False
    scaler: Type[str] = 'StandardScaler'
    scaler_model: Optional[str] = None
    root: str = ""
    model_type: str = "Hr"
    scaler_name: str = "scaler"

class Normalizer:
    def __init__(self,arguments,loader):
        self.params = arguments
        self.loader = loader


    def CreateBatch(self,targets=None,additionals=None,istarget=False):
        if not istarget:
            if isinstance(self.params.additionals,list) or isinstance(self.params.additionals,omegaconf.listconfig.ListConfig):
                Hr        = additionals.float().view(self.params.batch_size,len(self.params.additionals)).cpu().numpy()
            else:
                Hr = torch.Tensor([float(i[0]) for i in additionals]).view(self.params.batch_size,1).cpu().numpy()
        else:
            if isinstance(self.params.targets,list) or isinstance(self.params.targets,omegaconf.listconfig.ListConfig):
                Hr        = targets.float().view(self.params.batch_size,len(self.params.targets)).cpu().numpy()
            else:
                Hr = torch.Tensor([float(i[0]) for i in targets]).view(self.params.batch_size,1).cpu().numpy()
        
        return Hr

    def AddtoTotal(self,Hr,TrainHr):
        if TrainHr == []:
            TrainHr = Hr
        else:
            TrainHr = np.concatenate((TrainHr,Hr),axis=0)
        return TrainHr

    def ReShape(self,TrainHr):
        if isinstance(self.params.additionals,list) or isinstance(self.params.additionals,omegaconf.listconfig.ListConfig):
            TrainHr = TrainHr.reshape(-1,len(self.params.additional))
        else:
            TrainHr = TrainHr.reshape(-1,1)
        return TrainHr

    def LoadItems(self,loader,istarget=False):
        TrainHr = np.array([])
        for item in tqdm(loader, total=len(loader), smoothing=0.9):
            if self.params.hasaddons:
                if self.params.additionals is not None:
                    if self.params.graph == 'molecule': 
                        id,rtypes,Rgs,smiles,targets,additionals,Radd= item
                    elif self.params.graph == 'reaction':
                        id,rtypes,Rgs,Pgs,smiles,targets,additionals,Radd,Padd = item        
            else:
                if self.params.additionals is not None:
                    if self.params.graph == 'molecule':
                        id,rtypes,Rgs,smiles,targets,additionals = item
                    elif self.params.graph == 'reaction':
                        id,rtypes,Rgs,Pgs,smiles,targets,additionals = item
        
            Hr = self.CreateBatch(targets=targets,additionals=additionals,istarget=istarget)
            TrainHr = self.AddtoTotal(Hr,TrainHr)

            del id
            del rtypes
            del Rgs
            del smiles
            del targets
            if self.params.additionals is not None:del additionals
            if self.params.hasaddons: del Radd

            if not self.params.molecular:
                del Pgs
                if self.params.hasaddons: del Padd
            
        TrainHr = self.ReShape(TrainHr)
            
        return TrainHr

    def CreateScaler(self,TrainHr):
        scaler = getattr(BaseEstimator,self.params.scaler)()
        scaler.fit(TrainHr)
        scaled_res = scaler.transform(TrainHr)
        joblib.dump(scaler,f'{self.params.root}/{self.params.scaler_name}.pkl')
        return scaler,scaled_res
    
    def LoadScaler(self,TrainHr):
        scaler = joblib.load(self.params.scaler_model)
        scaled_res = scaler.transform(TrainHr)
        return scaler,scaled_res

    def TransformScaler(self,scaler,TrainHr):
        scaled_res = scaler.transform(TrainHr)
        return scaler,scaled_res
    
    def normalize(self):
        if self.params.model_type in ['Hr','Hr_multi']:
            if self.params.normtarget: 
                TrainHr = self.LoadItems(self.loader,istarget=False)
            else:
                TrainHr = self.LoadItems(self.loader,istarget=True)
        else:
            TrainHr = self.LoadItems(self.loader,istarget=True)
        
        if self.params.scaler_model is None:
            scaler,scaled_res = self.CreateScaler(TrainHr)
        else:
            scaler,scaled_res = self.LoadScaler(TrainHr)
        return scaler,scaled_res
    
    def normalize_with_loader(self,loader):
        if self.params.model_type in ['Hr','Hr_multi']:
            if self.params.normtarget: 
                TrainHr = self.LoadItems(loader,istarget=False)
            else:
                TrainHr = self.LoadItems(loader,istarget=True)
        else:
            TrainHr = self.LoadItems(loader,istarget=True)
        
        if self.params.scaler_model is None:
            scaler,scaled_res = self.CreateScaler(TrainHr)
        else:
            scaler,scaled_res = self.LoadScaler(TrainHr)
        return scaled_res
    
    def normalize_data(self,TrainHr):
        if self.params.scaler_model is None:
            scaler,scaled_res = self.CreateScaler(TrainHr)
        else:
            scaler,scaled_res = self.LoadScaler(TrainHr)
        return scaled_res

    def inverse(self,scaled_res):
        if self.params.scaler_model is None:
            res = scaler.inverse_transform(scaled_res)
            return res
        else:
            scaler = joblib.load(self.params.scaler_model)
            res = scaler.inverse_transform(scaled_res)
            return res
        