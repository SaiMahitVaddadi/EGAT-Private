from .numeric import NumericTensorFunctions
from .graphtocuda import DGLtoCUDA
from .obtainpreds import ObtainPreds
from tqdm import tqdm
import numpy as np

class BatchIteration(NumericTensorFunctions,DGLtoCUDA,ObtainPreds):
    def __init__(self, arguments):
        self.params = arguments

    def setupiteration(self):
        loss_list = []
        embeddingsdata = []
        data = []
        metrics_list = []
        return loss_list,embeddingsdata,data,metrics_list

    def anothercheckfornone(self,RGgs=None,PGgs=None,RAdd=None,PAdd=None):
        RGgs = RGgs if 'RGgs' in locals() else None
        PGgs = PGgs if 'PGgs' in locals() else None
        RAdd = RAdd if 'RAdd' in locals() else None
        PAdd = PAdd if 'PAdd' in locals() else None 
        return RGgs,PGgs,RAdd,PAdd
    
    def grabandmigrate(self,targets,Rgs,Pgs,Radd,Padd,additionals):
        targets = self.GrabTargets(targets)
        if self.params.model_type in ['BEP','Hr','Hr_multi','BEP_Hr']: additionals = self.GrabAdditionals(additionals)
        self.anothercheckfornone(Rgs,Pgs,Radd,Padd)
        if self.params.fingerprint == False: RGgs,PGgs,RAdd,PAdd = self.LoadGraphstoCUDA(Rgs,Pgs,Radd,Padd)
        else: Radd,Padd = self.LoadAddonstoCUDA(Radd,Padd)
        return targets,Rgs,Pgs,Radd,Padd,additionals
    
    def resetmlmodules(self,mode,loss):
        if mode == 'train': 
            loss.backward()
            self.optimizer.step()

    def appendlists(self,Rgs,Pgs,Radd,Padd,additionals,data,targets,smiles,id,rtypes,embeddingsdata,loss_list,metrics_list,mode):
        pred,embeddings,Rmap,Pmap = self.GetPrediction(Rgs,Pgs,Radd,Padd,additionals)
        datum,embeddingsdataum,loss,metrics = self.AddtoDataset(data,pred,targets,smiles,id,rtypes,embeddingsdata,embeddings)
        loss_list.append(loss.cpu().data.numpy())
        if self.params.metric is not None:  metrics_list.append(metrics.cpu().data.numpy())
        data.append(datum)
        embeddingsdata.append(embeddingsdataum)
        train_instance_acc = np.mean(loss_list)
        self.logger.info(f'{mode} accuracy is: %.5f' % train_instance_acc)
        return data,embeddingsdata,loss_list,metrics_list


    def batchiteration(self,item,data,embeddingsdata,loss_list,metrics_list,mode='train'):
        if self.params.fingerprint == False: 
            if 'reaction' in self.params.graph: 
                if self.params.additional != None:
                    if self.params.addons != None: 
                        id,rtypes,Rgs,Pgs,smiles,targets,additionals,Radd,Padd = item
                    else:
                        id,rtypes,Rgs,Pgs,smiles,targets,additionals = item
                else:
                    if self.params.addons != None: 
                        id,rtypes,Rgs,smiles,targets,Radd = item
                    else:
                        id,rtypes,Rgs,Pgs,smiles,targets = item
            elif 'molecular' in self.params.graph:
                if self.params.additional != None:
                    if self.params.addons != None: 
                        id,rtypes,Rgs,smiles,targets,additionals,Radd = item
                    else:
                        id,rtypes,Rgs,smiles,targets,additionals = item
                else:
                    if self.params.addons != None: 
                        id,rtypes,Rgs,smiles,targets,Radd = item
                    else:
                        id,rtypes,Rgs,smiles,targets = item
        else:
            if 'reaction' in self.params.graph: 
                if self.params.addons != None: 
                    if self.params.additional != None:
                        id,rtypes,smiles,targets,additionals,Radd,Padd = item
                    else:
                        id,rtypes,smiles,targets,Radd,Padd = item
            elif 'molecular' in self.params.graph:
                if self.params.addons != None: 
                    if self.params.additional != None:
                        id,rtypes,smiles,targets,additionals,Radd = item
                    else:
                        id,rtypes,smiles,targets,Radd = item
        targets,Rgs,Pgs,Radd,Padd,additionals = self.grabandmigrate(targets,Rgs,Pgs,Radd,Padd,additionals)
        if mode == 'train': self.optimizer.zero_grad()
        data,embeddingsdata,loss_list,metrics_list = self.appendlists(Rgs,Pgs,Radd,Padd,additionals,data,targets,smiles,id,rtypes,embeddingsdata,loss_list,metrics_list,mode)
        return data,embeddingsdata,loss_list,metrics_list
            


    def Iterate(self,loader,mode = 'train'):
        loss_list,embeddingsdata,data,metrics_list = self.setupiteration()
        for item in tqdm(loader, total=len(loader), smoothing=0.9):
            data,embeddingsdata,loss_list,metrics_list = self.batchiteration(item,data,embeddingsdata,loss_list,metrics_list,mode)
        return data,embeddingsdata,loss_list,metrics_list