import torch,logging,os,sys,hydra,omegaconf,shutil,importlib
from torch import nn
from .ml.setup import Setup
from ..dataset.Loader import EGATDataLoader
from .scaler import Scaler
import numpy as np 
from tqdm import tqdm
import pandas as pd 
from dataclasses import dataclass
from typing import List, Union, Optional



@dataclass
class PredictParams:
    molecular: bool
    target: Union[str, List[str]]
    batch_size: int
    model_type: str
    additionals: Optional[Union[str, List[str]]] = None
    Norm: Optional[str] = None
    hasaddons: bool = False
    Embed: bool = False
    AttnMaps: bool = False
    tweights: Optional[List[float]] = None
    save_path: str


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum
    return m

class Predict(Setup):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments
    
    def LoadData(self):
        self.data = EGATDataLoader(self.params)

    def LoadScaler(self):
        self.scaler = Scaler(self.params)
        self.scaler.normalize(self.data)

    def CreateCSV(self):
        if self.params.graph == 'reaction':
            csv = []
            columns = ['ID','RTYPE','Rsmiles','Psmiles','Rinchi','Pinchi']
            if isinstance(self.params.target,list):
                preds  = [t+'_PRED' for t in self.params.target]
                columns += preds        
                columns += self.params.target
            else:
                columns += [self.params.target+'_PRED']
                columns += [self.params.target]
        elif self.params.graph == 'molecular':
            csv = []
            columns = ['ID','RTYPE','Rsmiles','Rinchi']
            if isinstance(self.params.target,list):
                preds  = [t+'_PRED' for t in self.params.target]
                columns += preds        
                columns += self.params.target
            else:
                columns += [self.params.target+'_PRED']
                columns += [self.params.target]
        return csv,columns

    def ScaleData(self,scaler,additionals):
        if isinstance(self.params.additionals,list):
            Hr        = additionals.float().view(self.params.batch_size,len(self.params.additionals)).numpy()
        else:
            Hr        = additionals.float().view(self.params.batch_size,1).numpy()
        Hr        = scaler.transform(Hr).to(self.device)
        return Hr

    def GrabTargets(self,targets):
        if self.params.model_type in ['direct','BEP','Hr']:
            target    = torch.Tensor([float(i[2]) for i in targets]).view(self.params.batch_size,1).to(self.device)
        elif self.params.model_type in ['multi','Hr_multi']:
            target = targets.float().view(self.params.batch_size,len(self.params.target)).to(self.device)
        return target 
    
    def GetHr(self,additionals):
        if self.params.Norm is not None:
            Hr = self.ScaleData(self.scaler,additionals)
        else:
            if isinstance(self.params.additionals,list):
                Hr = additionals.float().view(self.params.batch_size,len(self.params.additionals)).to(self.device)
            else:
                Hr = torch.Tensor([float(i[0]) for i in additionals]).view(self.params.batch_size,1).to(self.device)
        return Hr


    def GrabAdditionals(self,additionals):
        if self.params.model_type == 'BEP':    
            if isinstance(self.params.additionals,list):
                raise ValueError('Error: BEP-like Prediction can only be done on one additional set of values.')
            else:
                Hr = torch.Tensor([float(i[0]) for i in additionals]).view(self.params.batch_size,1).to(self.device)
        elif self.params.model_type == 'Hr' or self.params.model_type == 'Hr_multi':
            Hr = self.GetHr(additionals)
        
        return Hr
    

    def LoadGraphstoCUDA(self,Rgs,Pgs,Radd,Padd):
        ##### TAKE THE GRAPHS AND LOAD THEM INTO CUDA
        RGgs      = Rgs.to(self.device)
        RGgs.ndata['x'].to(self.device)
        RGgs.edata['x'].to(self.device)
        if self.params.hasaddons: 
            RAdd = Radd.to(self.device)
        if self.params.graph == 'reaction':
            PGgs      = Pgs.to(self.device)
            PGgs.ndata['x'].to(self.device)
            PGgs.edata['x'].to(self.device)
            if self.params.hasaddons: 
                PAdd = Padd.to(self.device)
        RGgs = RGgs if 'RGgs' in locals() else None
        PGgs = PGgs if 'PGgs' in locals() else None
        RAdd = RAdd if 'RAdd' in locals() else None
        PAdd = PAdd if 'PAdd' in locals() else None
            
        return RGgs,PGgs,RAdd,PAdd
    

    def GetPrediction(self,RGgs,PGgs,RAdd,PAdd,Hr):
        ###### GET PREDICTION
        if self.params.model_type in ['direct','BEP','multi']:
            if self.params.graph == 'reaction':
                if self.params.hasaddons:
                    pred = self.model(RGgs,PGgs,RAdd,PAdd)
                else:
                    pred = self.model(RGgs,PGgs)
            elif self.params.graph == 'molecular':
                if self.params.hasaddons:
                    pred = self.model(RGgs,RAdd)
                else:
                    pred = self.model(RGgs)
        elif self.params.model_type in ['Hr','Hr_multi']:
            if self.params.graph == 'reaction':
                if self.params.hasaddons:
                    pred = self.model(RGgs,PGgs,Hr,RAdd,PAdd)
                else:
                    pred = self.model(RGgs,PGgs,Hr)
            elif self.params.graph == 'molecular':
                if self.params.hasaddons:
                    pred = self.model(RGgs,Hr,RAdd)
                else:
                    pred = self.model(RGgs,Hr)


        if not self.params.Embed:
            if self.params.AttnMaps:
                if self.params.graph == 'reaction':
                    pred,Rmap,Pmap = pred
                elif self.params.graph == 'molecular':
                    pred,Rmap = pred
        else:
            if self.params.AttnMaps:
                if self.params.graph == 'reaction':
                    pred,embeddings,Rmap,Pmap = pred
                elif self.params.graph == 'molecular':
                    pred,embeddings,Rmap = pred
            else:
                pred,embeddings = pred
                

        if self.params.model_type == 'BEP': pred = pred[:, 0].unsqueeze(1) * Hr + pred[:, 1].unsqueeze(1)


        pred = pred if 'RGgs' in locals() else None
        embeddings = embeddings if 'PGgs' in locals() else None
        Rmap = Rmap if 'RAdd' in locals() else None
        Pmap = Pmap if 'PAdd' in locals() else None    
        return pred,embeddings,Rmap,Pmap
    
    def AddtoDataset(self,data,pred,target,smiles,id,rtypes,embeddingsdata,embeddings):
        # Merge the torch Tensors, get them out of cuda, and move them to numpy 
        batch_result = torch.cat([pred,target]).cpu().data
        batch_result = torch.cat([smiles,batch_result]).numpy()
        batch_result = np.hstack((id,rtypes,batch_result)).tolist()
        if self.params.Embed: embeddingsdata.append(embeddings.cpu().numpy().tolist())
        data.append(batch_result)

        if self.params.model_type in ['Hr','Hr_multi']:
            if len(self.params.tweights) == len(self.params.targets):
                loss = None
                for index,weight in enumerate(self.params.tweights):
                    p = pred[:,index].unsqueeze(1)
                    t = target[:,index].unsqueeze(1)
                    if loss is not None:
                        loss += self.params.tweights[index] * self.loss(p,t)
                    else:
                        loss = self.params.tweights[index] * self.loss(p,t)
            else:
                raise ValueError('Cannot work because the weights are underdetermined.')
        else:
            loss = self.loss(pred, target)
        return data,embeddingsdata,loss

            
    def Iterate(self,loader,mode = 'train'):
        loss_list = []
        embeddingsdata = []
        data = []
        for item in tqdm(loader, total=len(loader), smoothing=0.9):
            if self.params.hasaddons:
                if self.params.additionals is not None:
                    if self.params.graph == 'molecule': 
                        id,rtypes,Rgs,smiles,targets,additionals,Radd= item
                    elif self.params.graph == 'reaction':
                        #collateall
                        id,rtypes,Rgs,Pgs,smiles,targets,additionals,Radd,Padd = item
                else:
                    if self.params.model_type in ['Hr','BEP','Hr_multi']:
                        self.logger.info('Error: Predictions Require Additional Values that are not given.')
                        break 
                    else:
                        if self.params.graph == 'molecule': 
                            id,rtypes,Rgs,smiles,targets,Radd= item
                        elif self.params.graph == 'reaction':
                            id,rtypes,Rgs,smiles,targets,Radd = item
            else:
                if self.params.additionals is not None:
                    if self.params.graph == 'molecule':
                        id,rtypes,Rgs,smiles,targets,additionals = item
                    elif self.params.graph == 'reaction':
                        id,rtypes,Rgs,Pgs,smiles,targets,additionals = item
                else:
                    if self.params.model_type in ['Hr','BEP','Hr_multi']:
                        self.logger.info('Error: Predictions Require Additional Values that are not given.')
                        break 
                    else:
                        if self.params.graph == 'molecule':
                            id,rtypes,Rgs,Pgs,smiles,targets = item
                        elif self.params.graph == 'reaction':
                            id,rtypes,Rgs,smiles,targets = item
            
            targets = self.GrabTargets(targets)
            if self.params.model_type in ['BEP','Hr','Hr_multi']: additionals = self.GrabAdditionals(additionals)
            
            RGgs = RGgs if 'RGgs' in locals() else None
            PGgs = PGgs if 'PGgs' in locals() else None
            RAdd = RAdd if 'RAdd' in locals() else None
            PAdd = PAdd if 'PAdd' in locals() else None    
            RGgs,PGgs,RAdd,PAdd = self.LoadGraphstoCUDA(Rgs,Pgs,Radd,Padd)

            if mode == 'train': self.optimizer.zero_grad()
            pred,embeddings,Rmap,Pmap = self.GetPrediction(RGgs,PGgs,RAdd,PAdd,additionals)
            datum,embeddingsdataum,loss = self.AddtoDataset(data,pred,targets,smiles,id,rtypes,embeddingsdata,embeddings)

            if mode == 'train': loss.backward()
            loss_list.append(loss.cpu().data.numpy())
            data.append(datum)
            embeddingsdata.append(embeddingsdataum)
            
            if mode == 'train': self.optimizer.step()
        
        train_instance_acc = np.mean(loss_list)
        self.logger.info(f'{mode} accuracy is: %.5f' % train_instance_acc)

        return data,embeddingsdata,loss_list

    def Loop(self):
        '''learning one epoch'''
        ###### LOAD THE PREDICTION DATAFRAME AND ITS COLUMNS
        data,columns = self.CreateCSV()
        self.logger.info('Training...')
        with torch.no_grad():
            self.model = self.model.eval()
            data,embeddings,loss_list = self.Iterate(self.data['val'],mode = 'test')
        self.SaveData(data,columns)
        if embeddings: self.SaveEmbeddings(embeddings)

        return loss_list
    
    
    def SaveData(self,data,columns):
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(self.params.save_path)

    def SaveEmbeddings(self,embeddings):
        df = pd.DataFrame(embeddings)
        df.to_csv(self.params.save_path.split('.')[0]+'_embeddings.csv')
        
    def Run(self):  
        loss_list = self.Loop()
        avg_loss = np.mean(loss_list)
        self.logger.info(f'Average loss: {avg_loss:.5f}')

    