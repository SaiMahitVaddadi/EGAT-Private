import torch,logging,os,sys,hydra,omegaconf,shutil,importlib
from torch import nn
from .setup import Setup
from ..loader.Loader import EGATDataLoader
from .scaler import Scaler
import numpy as np 
from tqdm import tqdm
import pandas as pd 
from dataclasses import dataclass, field
from typing import List, Optional, Union



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

    




def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum
    return m

class Train(Setup):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments
        self.LoadData()
        if self.params.norm is not None: self.LoadScaler()
    
    def LoadData(self):
        self.data = EGATDataLoader(self.params)


    def CombinedLoader(self):
        if self.params.scale_region == 'all':
            if self.params.test_only: dataset = [self.data['train'].dataset, self.data['val'].dataset]
            else: dataset = [self.data['train'].dataset, self.data['val'].dataset,self.data['test'].dataset]
        elif self.params.scale_region == 'trainval' or self.params.scale_region == 'valtrain':
            dataset = [self.data['train'].dataset, self.data['val'].dataset]
        elif self.params.scale_region == 'valtest' or self.params.scale_region == 'testval':
            dataset = [self.data['val'].dataset, self.data['test'].dataset]
        elif self.params.scale_region == 'traintest' or self.params.scale_region == 'testtrain':
            dataset = [self.data['train'].dataset, self.data['test'].dataset]
        combined_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(dataset),
            batch_size=self.params.batch_size,
            shuffle=self.params.randomize,
            num_workers=self.params.num_workers
        )
        return combined_loader

    def LoadScaler(self):
        self.scaler = Scaler(self.params)
        combinedargs = ['all','trainval','traintest','testtrain','valtrain','valtest','testval']        
        noncombinedargs = ['train','val','test']
        
        if self.params.scale_region in combinedargs:
            loader = self.CombinedLoader()
        elif self.params.scale_region in noncombinedargs:
            loader = self.data[self.params.scale_region]
        self.normalizer = dict()
        self.normalizer['model'],self.normalizer['result'] = self.scaler.normalize




    def LoadLearningRate(self,epoch):
        if epoch < self.params.epoch:
            if self.params.scheduler == 'step':
                lr = max(self.params.learning_rate * (self.params.lr_decay ** (epoch // self.params.step_size)), self.LEARNING_RATE_CLIP)
            elif self.params.scheduler == 'exp':
                if epoch < self.params.warmup:
                    lr_factor = (epoch + 1) * 1.0 / self.params.warmup
                else:
                    lr_factor = np.exp( - self.params.expdecay * (epoch - self.params.warmup + 1) / self.params.epoch)
                lr = max(self.params.learning_rate * lr_factor, self.LEARNING_RATE_CLIP)
            elif self.params.scheduler == 'cos':
                if epoch < self.params.warmup:
                    lr = self.params.learning_rate * (epoch + 1) * 1.0 / self.params.warmup
                else:
                    lr = self.LEARNING_RATE_CLIP + 0.5 * (self.params.learning_rate - self.LEARNING_RATE_CLIP) * (1 + np.cos(np.pi * (epoch - self.params.warmup + 1) / (self.params.epoch - self.params.warmup + 1)))
        else:
            lr = self.LEARNING_RATE_CLIP

    def UpdateLearningRate(self,lr):
        self.logger.info('Learning rate:%f' % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def UpdateMomentum(self,epoch):
        momentum = self.MOMENTUM_ORIGINAL * (self.MOMENTUM_DECAY ** (epoch //self.MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        return momentum
    
    def CreateCSV(self):
        if not self.params.molecular:
            csv = []
            columns = ['ID','RTYPE','Rsmiles','Psmiles','Rinchi','Pinchi']
            if isinstance(self.params.target,list):
                preds  = [t+'_PRED' for t in self.params.target]
                columns += preds        
                columns += self.params.target
            else:
                columns += [self.params.target+'_PRED']
                columns += [self.params.target]
        else:
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
        
        self.mask = ~torch.isnan(target)
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
        if not self.params.molecular:
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
            if not self.params.molecular:
                if self.params.hasaddons:
                    pred = self.model(RGgs,PGgs,RAdd,PAdd)
                else:
                    pred = self.model(RGgs,PGgs)
            else:
                if self.params.hasaddons:
                    pred = self.model(RGgs,RAdd)
                else:
                    pred = self.model(RGgs)
        elif self.params.model_type in ['Hr','Hr_multi']:
            if self.params.norm: Hr = self.ScaleData(Hr)
            if not self.params.molecular:
                if self.params.hasaddons:
                    pred = self.model(RGgs,PGgs,Hr,RAdd,PAdd)
                else:
                    pred = self.model(RGgs,PGgs,Hr)
            else:
                if self.params.hasaddons:
                    pred = self.model(RGgs,Hr,RAdd)
                else:
                    pred = self.model(RGgs,Hr)

        
        if not self.params.Embed:
            if self.params.AttnMaps:
                if not self.params.molecular:
                    pred,Rmap,Pmap = pred
                else:
                    pred,Rmap = pred
        else:
            if self.params.AttnMaps:
                if not self.params.molecular:
                    pred,embeddings,Rmap,Pmap = pred
                else:
                    pred,embeddings,Rmap = pred
            else:
                pred,embeddings = pred
                

        if self.params.model_type == 'BEP': pred = pred[:, 0].unsqueeze(1) * Hr + pred[:, 1].unsqueeze(1)

        pred = pred if 'RGgs' in locals() else None
        embeddings = embeddings if 'PGgs' in locals() else None
        Rmap = Rmap if 'RAdd' in locals() else None
        Pmap = Pmap if 'PAdd' in locals() else None    
        return pred,embeddings,Rmap,Pmap
    


    def WeightedLoss(self,pred,target):
        if len(self.params.tweights) == len(self.params.targets):
            loss = None
            for index,weight in enumerate(self.params.tweights):
                p = pred[:,index].unsqueeze(1)
                t = target[:,index].unsqueeze(1)
                if loss is not None:
                    loss += self.params.tweights[index] * self.GetLoss(p,t)
                else:
                    loss = self.params.tweights[index] * self.GetLoss(p,t)
        else:
            raise ValueError('Cannot work because the weights are underdetermined.')
        return loss 

    def WeightedMetric(self,pred,target):
        if len(self.params.tweights) == len(self.params.targets):
            loss = None
            for index,weight in enumerate(self.params.tweights):
                p = pred[:,index].unsqueeze(1)
                t = target[:,index].unsqueeze(1)
                if loss is not None:
                    loss += self.params.tweights[index] * self.GetMetric(p,t)
                else:
                    loss = self.params.tweights[index] * self.GetMetric(p,t)
        else:
            raise ValueError('Cannot work because the weights are underdetermined.')
        return loss 
    
    def GetAllMetrics(self,pred,target):
        metrics = []
        for index, weight in enumerate(self.params.tweights):
            p = pred[:, index].unsqueeze(1)
            t = target[:, index].unsqueeze(1)
            metric = self.GetMetric(p, t)
            metrics.append(metric)
        return metrics

    def ScaleData(self,data):
        return self.normalizer['model'].normalize_data(data)  

    def InverseData(self,data):
        return self.normalizer['model'].inverse(data)        

    def AddtoDataset(self,data,pred,target,smiles,id,rtypes,embeddingsdata,embeddings):
        # Merge the torch Tensors, get them out of cuda, and move them to numpy 
        batch_result = torch.cat([pred,target]).cpu().data
        batch_result = torch.cat([smiles,batch_result]).numpy()
        batch_result = np.hstack((id,rtypes,batch_result)).tolist()
        if self.params.Embed: embeddingsdata.append(embeddings.cpu().numpy().tolist())
        data.append(batch_result)


        if self.params.normtarget:
            if not self.params.trainedonnorm:
                pred = self.ScaleData(pred)
                target = self.ScaleData(target)
            else:
                pred = self.InverseData(pred)
                target = self.InverseData(target)

        if self.params.model_type in ['multi','Hr_multi']:
            loss = self.WeightedLoss(pred,target)
            if self.params.metric is not None: 
                comb_metric = self.WeightedMetric(pred,target)
                metric = self.GetAllMetrics(pred,target)
            else:
                comb_metric = None
        else:
            self.GetLoss(pred,target)
            comb_metric = None
            if self.params.metric is not None: metric = self.GetMetric(pred,target)
            else: metric = None 
        return data,embeddingsdata,loss,comb_metric,metric


    def ArithMean(self, pred, target, weights=None):
        loss = 0
        if weights is None:
            weights = [1] * len(self.loss)
        for i, lossfunc in enumerate(self.loss):
            try:
                loss += weights[i] * lossfunc(pred, target)
            except:
                loss += weights[i] * lossfunc(pred[self.mask], target[self.mask])
        return loss / sum(weights)
    
    def GeomMean(self, pred, target, weights=None):
        loss = 1
        if weights is None:
            weights = [1] * len(self.loss)
        for i, lossfunc in enumerate(self.loss):
            try:
                loss *= (lossfunc(pred, target) ** weights[i])
            except:
                loss *= (lossfunc(pred[self.mask], target[self.mask]) ** weights[i])
        return loss ** (1 / sum(weights))

    def HarmMean(self, pred, target, weights=None):
        loss = 0
        if weights is None:
            weights = [1] * len(self.loss)
        for i, lossfunc in enumerate(self.loss):
            try:
                loss += weights[i] / lossfunc(pred, target)
            except:
                loss += weights[i] / lossfunc(pred[self.mask], target[self.mask])
        return sum(weights) / loss

    def QuadMean(self, pred, target, weights=None):
        loss = 0
        if weights is None:
            weights = [1] * len(self.loss)
        for i, lossfunc in enumerate(self.loss):
            try:
                loss += weights[i] * (lossfunc(pred, target) ** 2)
            except:
                loss += weights[i] * (lossfunc(pred[self.mask], target[self.mask]) ** 2)
        return (loss / sum(weights)) ** 0.5
    

    def GetLoss(self,pred,target):
        if isinstance(self.loss,nn.Module):
            try:
                loss = self.loss(pred, target)
            except:
                loss = self.loss(pred[self.mask], target[self.mask])
        elif isinstance(self.loss,list):
            loss = getattr(f'{self.params.loss_agg}Mean')
            loss = loss(pred, target, weights=self.params.loss_weights)
        return loss

    

    def Iterate(self,loader,mode = 'train'):
        loss_list = []
        embeddingsdata = []
        data = []
        metrics_list = []
        comb_metrics = []
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
            datum,embeddingsdataum,loss,comb_metrics,metrics = self.AddtoDataset(data,pred,targets,smiles,id,rtypes,embeddingsdata,embeddings)

            if mode == 'train': loss.backward()
            loss_list.append(loss.cpu().data.numpy())
            if self.params.metric is not None: 
                metrics_list.append(metrics.cpu().data.numpy())
                if comb_metrics is not None:
                    comb_metrics.append(comb_metrics.cpu().data.numpy())
            data.append(datum)
            embeddingsdata.append(embeddingsdataum)
            
            if mode == 'train': self.optimizer.step()
        
        train_instance_acc = np.mean(loss_list)
        self.logger.info(f'{mode} accuracy is: %.5f' % train_instance_acc)

        return data,embeddingsdata,loss_list,metrics_list,comb_metrics

    def Loop(self,epoch):
        '''Adjust learning rate and BN momentum'''
        # set up lr
        lr = self.LoadLearningRate(epoch)
        self.UpdateLearningRate(lr)        

        # update momentum
        momentum = self.UpdateMomentum(epoch)
        
        self.logger.info('BN momentum updated to: %f' % momentum)
        self.model = self.model.apply(lambda x: bn_momentum_adjust(x, momentum))
        self.model = self.model.train()

        '''learning one epoch'''
        ###### LOAD THE PREDICTION DATAFRAME AND ITS COLUMNS
        train,self.columns = self.CreateCSV()
        val,valcolumns = self.CreateCSV()
        test,testcolumns = self.CreateCSV()
        self.logger.info('Training...')
        train,train_embeddings,train_loss_list,train_metrics_list,train_comb_metrics_list = self.Iterate(self.data['train'],mode = 'train')
        with torch.no_grad():
            self.model = self.model.eval()
            val,val_embeddings,val_loss_list,val_metrics_list,val_comb_metrics_list = self.Iterate(self.data['val'],mode = 'test')
            if not self.params.test_only: 
                test,test_embeddings,test_loss_list,test_metrics_list,test_comb_metrics_list = self.Iterate(self.data['test'],mode = 'test')
            else:
                test = None
                test_embeddings = None
                test_loss_list = None
        
        return train,val,test,train_embeddings,val_embeddings,test_embeddings,train_loss_list,val_loss_list,test_loss_list,lr,train_metrics_list,val_metrics_list,test_metrics_list,train_comb_metrics_list,val_comb_metrics_list,test_comb_metrics_list
        

    def UpdateWandB(self,train,test,val,epoch,lr):
        # compute the average
        if self.params.weightsandbiases:
            if self.params.test_only:
                wandb.log({"learning_rate":lr, "train_loss": np.mean(train), 'val_loss': np.mean(val)},step=epoch)
            else:
                wandb.log({"learning_rate":lr, "train_loss": np.mean(train), 'val_loss': np.mean(val),'ext_loss': np.mean(test)},step=epoch)


    def SaveTorchModel(self,train,val,test,train_embeddings,val_embeddings,test_embeddings,val_loss_list,train_loss_list,epoch,columns,ensemble=None,fold=None):
        if np.mean(val_loss_list) < self.best_loss:  
            self.loss_increase_count = 0
            self.best_loss = np.mean(val_loss_list)
            self.logger.info('Save model...')
            
            if ensemble is not None:
                if fold is not None:
                    savepath = f'best_model_ensemble_{ensemble}_fold_{fold}.pth'
                else:
                    savepath = f'best_model_ensemble_{ensemble}.pth'
            else:
                if fold is not None:
                    savepath = f'best_model_fold_{fold}.pth'
                else:
                    savepath = 'best_model.pth'

            self.logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': np.mean(train_loss_list),
                'test_acc': self.best_loss,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(state, savepath)
            self.logger.info('Saving model....')
            if self.params.save_style == 'best':
                self.SaveData(train,val,test,columns,ensemble,fold)
                self.SaveEmbeddings(train_embeddings,val_embeddings,test_embeddings,ensemble,fold)
        else:
            if 'every-' in self.params.save_style:    
                interval = int(self.params.save_style.split('-')[1])
                if epoch % interval == 0:
                    self.logger.info('Save model...')
                    if ensemble is not None:
                        if fold is not None:
                            savepath = f'epoch_{epoch}_ensemble_{ensemble}_fold_{fold}.pth'
                        else:
                            savepath = f'epoch_{epoch}_ensemble_{ensemble}.pth'
                    else:
                        if fold is not None:
                            savepath = f'epoch_{epoch}_fold_{fold}.pth'
                        else:
                            savepath = f'epoch_{epoch}.pth'
                    self.logger.info('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'train_acc': np.mean(train_loss_list),
                        'test_acc': np.mean(val_loss_list),
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    self.logger.info('Saving model....')
                    self.SaveDataAtEpoch(train,val,test,columns,epoch,ensemble,fold)
                    self.SaveEmbeddingsAtEpoch(train_embeddings,val_embeddings,test_embeddings,ensemble,fold)
               
    
    def SaveData(self,train,val,test,columns,ensemble=None,fold=None):
        ###### SAVE RESULTS AS CSV
        datasets = {'train': train, 'val': val, 'test': test}
        if ensemble is None:
            if fold is None:
                filenames = {'train': 'best_train.csv', 'val': 'best_val.csv', 'test': 'best_test.csv'}
            else:
                filenames = {'train': f'best_train_fold_{fold}.csv', 'val': f'best_val_fold_{fold}.csv', 'test': f'best_test_fold_{fold}.csv'}
        else:
            if fold is None:
                filenames = {'train': f'best_train_ensemble_{ensemble}.csv', 'val': f'best_val_ensemble_{ensemble}.csv', 'test': f'best_test_ensemble_{ensemble}.csv'}
            else:
                filenames = {'train': f'best_train_ensemble_{ensemble}_fold_{fold}.csv', 'val': f'best_val_ensemble_{ensemble}_fold_{fold}.csv', 'test': f'best_test_ensemble_{ensemble}_fold_{fold}.csv'}
            
        for key in datasets:
            if datasets[key] is not None:
                df = pd.DataFrame(datasets[key], columns=columns)
                df.to_csv(filenames[key])
        

    def SaveDataAtEpoch(self,train,val,test,columns,epoch=1,ensemble=None,fold=None):
        ###### SAVE RESULTS AS CSV
        datasets = {'train': train, 'val': val, 'test': test}
        if ensemble is None:
            if fold is None:
                filenames = {'train': f'epoch_{epoch}_train.csv', 'val': f'epoch_{epoch}_val.csv', 'test': f'epoch_{epoch}_test.csv'}
            else:
                filenames = {'train': f'epoch_{epoch}_train_fold_{fold}.csv', 'val': f'epoch_{epoch}_val_fold_{fold}.csv', 'test': f'epoch_{epoch}_test_fold_{fold}.csv'}
        else:
            if fold is None:
                filenames = {'train': f'epoch_{epoch}_train_ensemble_{ensemble}.csv', 'val': f'epoch_{epoch}_val_ensemble_{ensemble}.csv', 'test': f'epoch_{epoch}_test_ensemble_{ensemble}.csv'}
            else:
                filenames = {'train': f'epoch_{epoch}_train_ensemble_{ensemble}_fold_{fold}.csv', 'val': f'epoch_{epoch}_val_ensemble_{ensemble}_fold_{fold}.csv', 'test': f'epoch_{epoch}_test_ensemble_{ensemble}_fold_{fold}.csv'}

        for key in datasets:
            if datasets[key] is not None:
                df = pd.DataFrame(datasets[key], columns=columns)
                df.to_csv(filenames[key])

    def SaveEmbeddingsAtEpoch(self,train,val,test,epoch=1,ensemble=None,fold=None):
        ###### SAVE RESULTS AS CSV
        datasets = {'train': train, 'val': val, 'test': test}
        if ensemble is None:
            if fold is None:
                filenames = {'train': f'epoch_{epoch}_train_embeddings.csv', 'val': f'epoch_{epoch}_val_embeddings.csv', 'test': f'epoch_{epoch}_test_embeddings.csv'}
            else:
                filenames = {'train': f'epoch_{epoch}_train_embeddings_fold_{fold}.csv', 'val': f'epoch_{epoch}_val_embeddings_fold_{fold}.csv', 'test': f'epoch_{epoch}_test_embeddings_fold_{fold}.csv'}
        else:
            if fold is None:
                filenames = {'train': f'epoch_{epoch}_train_embeddings_ensemble_{ensemble}.csv', 'val': f'epoch_{epoch}_val_embeddings_ensemble_{ensemble}.csv', 'test': f'epoch_{epoch}_test_embeddings_ensemble_{ensemble}.csv'}
            else:
                filenames = {'train': f'epoch_{epoch}_train_embeddings_ensemble_{ensemble}_fold_{fold}.csv', 'val': f'epoch_{epoch}_val_embeddings_ensemble_{ensemble}_fold_{fold}.csv', 'test': f'epoch_{epoch}_test_embeddings_ensemble_{ensemble}_fold_{fold}.csv'}

        for key in datasets:
            if datasets[key] is not None:
                df = pd.DataFrame(datasets[key])
                df.to_csv(filenames[key])

    def SaveLossesToCSV(self, train_loss_list, val_loss_list, test_loss_list, lr, 
                        epoch,train_metrics_list,val_metrics_list,test_metrics_list,train_comb_metrics_list,
                        val_comb_metrics_list,test_comb_metrics_list,ensemble=None,fold=None):
        # Create a dictionary with the data
        data = {
            'epoch': [epoch],
            'learning_rate': [lr],
            'train_loss': [np.mean(train_loss_list)],
            'val_loss': [np.mean(val_loss_list)],
            'test_loss': [np.mean(test_loss_list) if test_loss_list is not None else None]
        }
        

        # Check if metrics lists are either None or a list of Nones
        def check_metrics_list(metrics_list):
            return metrics_list is None or all(metric is None for metric in metrics_list)

        if not check_metrics_list(train_metrics_list):
            data['train_metrics'] = [np.mean(train_metrics_list)]
        if not check_metrics_list(val_metrics_list):
            data['val_metrics'] = [np.mean(val_metrics_list)]
        if not check_metrics_list(test_metrics_list):
            data['test_metrics'] = [np.mean(test_metrics_list) if test_metrics_list is not None else None]
        if not check_metrics_list(train_comb_metrics_list):
            data['train_comb_metrics'] = [np.mean(train_comb_metrics_list)]
        if not check_metrics_list(val_comb_metrics_list):
            data['val_comb_metrics'] = [np.mean(val_comb_metrics_list)]
        if not check_metrics_list(test_comb_metrics_list):
            data['test_comb_metrics'] = [np.mean(test_comb_metrics_list) if test_comb_metrics_list is not None else None]

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)
        
        # Define the filename
        if ensemble is None:
            if fold is None:
                filename = 'losses.csv'
            else:
                filename = f'losses_fold_{fold}.csv'
        else:
            if fold is None:
                filename = f'losses_ensemble_{ensemble}.csv'
            else:
                filename = f'losses_ensemble_{ensemble}_fold_{fold}.csv'
        
        # Save the DataFrame to a CSV file
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)

    def SaveEmbeddings(self,train,val,test,ensemble=None,fold=None):
        ###### SAVE RESULTS AS CSV
        datasets = {'train': train, 'val': val, 'test': test}
        if ensemble is None:
            if fold is None:
                filenames = {'train': 'best_train_embeddings.csv', 'val': 'best_val_embeddings.csv', 'test': 'best_test_embeddings.csv'}
            else:
                filenames = {'train': f'best_train_embeddings_fold_{fold}.csv', 'val': f'best_val_embeddings_fold_{fold}.csv', 'test': f'best_test_embeddings_fold_{fold}.csv'}
        else:
            if fold is None:
                filenames = {'train': f'best_train_embeddings_ensemble_{ensemble}.csv', 'val': f'best_val_embeddings_ensemble_{ensemble}.csv', 'test': f'best_test_embeddings_ensemble_{ensemble}.csv'}
            else:
                filenames = {'train': f'best_train_embeddings_ensemble_{ensemble}_fold_{fold}.csv', 'val': f'best_val_embeddings_ensemble_{ensemble}_fold_{fold}.csv', 'test': f'best_test_embeddings_ensemble_{ensemble}_fold_{fold}.csv'}

        for key in datasets:
            if datasets[key] is not None:
                df = pd.DataFrame(datasets[key])
                df.to_csv(filenames[key])

    def TrainbyEpoch(self,ensemble=None,fold=None):
        for epoch in range(self.start_epoch, self.params.epoch+self.params.epoch_const):
            self.loss_increase_count += 1
            self.logger.info('Epoch %d (%d/%s):' % (self.global_epoch + 1, epoch + 1, self.params.epoch))
            train,val,test,train_embeddings,val_embeddings,test_embeddings,train_loss_list,val_loss_list,test_loss_list,lr,train_metrics_list,val_metrics_list,test_metrics_list,train_comb_metrics_list,val_comb_metrics_list,test_comb_metrics_list = self.Loop(epoch)
            self.UpdateWandB(train,val,test,train_embeddings,val_embeddings,test_embeddings,train_loss_list,val_loss_list,test_loss_list,lr)
            self.SaveTorchModel(train,val,test,train_embeddings,val_embeddings,test_embeddings,val_loss_list,train_loss_list,epoch,self.columns,ensemble,fold)
            self.SaveLossesToCSV(train_loss_list, val_loss_list, test_loss_list, lr, epoch,train_metrics_list,val_metrics_list,test_metrics_list,train_comb_metrics_list,val_comb_metrics_list,test_comb_metrics_list,ensemble,fold)
            if self.loss_increase_count > self.params.patience:
                break
            self.global_epoch += 1
        
        return np.mean(val_loss_list)

    
    def TrainUntilConvergence(self,ensemble=None,fold=None):
        prev_loss = float('inf')
        epoch = self.start_epoch
        while abs(prev_loss - current_loss) >= self.params.loss_threshold:
            self.logger.info('Epoch %d:' % (self.global_epoch + 1))
            train,val,test,train_embeddings,val_embeddings,test_embeddings,train_loss_list,val_loss_list,test_loss_list,lr,train_metrics_list,val_metrics_list,test_metrics_list,train_comb_metrics_list,val_comb_metrics_list,test_comb_metrics_list = self.Loop(epoch)
            self.UpdateWandB(train_loss_list, val_loss_list, test_loss_list, lr)
            self.SaveTorchModel(train, val, test, train_embeddings, val_embeddings, test_embeddings, val_loss_list, train_loss_list, epoch, self.columns, ensemble,fold)
            self.SaveLossesToCSV(train_loss_list, val_loss_list, test_loss_list, lr, epoch,train_metrics_list,val_metrics_list,test_metrics_list,train_comb_metrics_list,val_comb_metrics_list,test_comb_metrics_list,ensemble,fold)
            current_loss = np.mean(val_loss_list)
            if epoch == self.params.epoch + self.params.epoch_const:
                break
            prev_loss = current_loss
            self.global_epoch += 1
            epoch += 1
        
        return current_loss
    

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
    

    