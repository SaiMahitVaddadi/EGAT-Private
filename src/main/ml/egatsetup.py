import torch,logging,hydra
from torch import nn
import os
from ...models.base.egatmodel import EGATModel,MultiCompEGATModel
from ...models.base.fpmodel import FPModel,MultiComponentFPModel
from ...models.ablation.ablationmodels import MolecularAblationModel,ReactionAblationModel,MolecularAblationModelwAddOns,ReactionAblationModelwAddOns
from .finetuning import FinetuningEGAT
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from ...loader.Loader import EGATDataLoader
from copy import deepcopy

@dataclass
class EGATSetupParams:
    seed: Optional[int] = None
    gpu: int = 0
    save_path: str = "./"
    data_path: str = "./data"
    smiles: Optional[List[str]] = None
    trainer: str = "EGAT"
    model: str = "EGATModel"
    startpoint: str = "Retrain"
    base_model: Optional[str] = None
    ablation_EGAT_model: Optional[str] = None
    ablation_NN_model: Optional[str] = None
    graph: str = "molecule"
    hasaddons: bool = False
    parallel: bool = False
    finetuner: Optional[str] = None
    defaults: Optional[dict] = None


class EGATModelSetup:
    def __init__(self,arguments):
        self.params = arguments
        self.logger = logging.getLogger(__name__)

    def LoadData(self):
        self.dataloader = EGATDataLoader(self.params)

    def SetSeed(self):
        if self.params.seed is not None:
            self.logger.info(f'Setting random seed to {self.params.seed}')
            torch.manual_seed(self.params.seed)
            torch.cuda.manual_seed_all(self.params.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            random.seed(self.params.seed)
            np.random.seed(self.params.seed)
    
    def StartFromScratch(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.params.gpu) # Get the gpus
        self.logger.info('No existing model, starting training from scratch...')
        best_loss = 100
        start_epoch = 0
        checkpoint = None
        return best_loss, start_epoch,checkpoint
    
    def PreLoadModel(self,model=None):
        if model is None: model= os.path.join(self.params.save_path,'best_model.pth')
        checkpoint = torch.load(model) # Set model to path
        start_epoch = checkpoint['epoch'] # Get starting point 
        self.logger.info('Use pretrain model, starting from epoch {}'.format(start_epoch))
        best_loss = checkpoint['test_acc'] # Get best loss for reference
        return best_loss, start_epoch,checkpoint
    
    def LoadModelForPretraining(self,model=None):
        if model is None: model= os.path.join(self.params.save_path,'best_model.pth')
        checkpoint = torch.load(model)  # Set model to path
        self.logger.info('Pre-trained Model is loaded, starting training on current data from scratch...')
        best_loss = 100
        start_epoch = 0
        return best_loss, start_epoch,checkpoint
    
    def retraincase(self):
        try:
            best_loss, start_epoch,checkpoint = self.PreLoadModel()
        except:
            best_loss, start_epoch,checkpoint = self.StartFromScratch()

        return best_loss, start_epoch,checkpoint
    
    def pretraincase(self):
        try:
            best_loss, start_epoch,checkpoint = self.LoadModelForPretraining(self.params.base_model)
        except:
            best_loss, start_epoch,checkpoint = self.StartFromScratch()
        return best_loss, start_epoch,checkpoint
    
    def frankensteincase(self):
        try:
            best_loss, start_epoch,checkpointA = self.LoadModelForPretraining(self.params.ablation_EGAT_model)
            checkpoint = None
        except:
            best_loss, start_epoch,checkpoint = self.StartFromScratch()
            checkpointA = None

        try:
            best_loss, start_epoch,checkpointB = self.LoadModelForPretraining(self.params.ablation_NN_model)
            checkpoint = None
        except:
            best_loss, start_epoch,checkpoint = self.StartFromScratch()
            
            checkpointB = None
        
        return best_loss, start_epoch,checkpointA,checkpointB,checkpoint

    def LoadModel(self):
        if self.params.startpoint in ['Retrain','Predict']: 
            best_loss, start_epoch,checkpoint = self.retraincase()
            return best_loss, start_epoch,checkpoint
            
        ### Attempt to find a pre-trained model. If not, stop.
        elif self.params.startpoint in ['Pretrain','EGAT_Freeze','NN_Freeze','EGAT_Finetune','NN_Finetune']: 
            best_loss, start_epoch,checkpoint = self.pretraincase()    
            return best_loss, start_epoch,checkpoint           
            
        ### Attempt to find a pre-trained model and freeze the EGAT side
        elif self.params.startpoint == 'EGAT_Ablation' or self.params.startpoint == 'NN_Ablation': # Attempt to find two pre-trained models. If not, stop.
            best_loss, start_epoch,checkpointA,checkpointB,checkpoint = self.frankensteincase()
            return best_loss, start_epoch,checkpointA,checkpointB,checkpoint
        # Attempt to start from scratch.
        else: 
            best_loss, start_epoch,checkpoint = self.StartFromScratch()
            return best_loss, start_epoch,checkpoint

    def InitializeData(self):
        '''INITALIIZATION of DATA'''
        ### Get the data path, otherwise stop if it's not found. 
        try:
            root = hydra.utils.to_absolute_path(self.params.data_path) # Get the data path
            return True
        except:
            self.logger.info('Data Path cannot be found')
            return None    

    def LoadPredictor(self,num_node_feats,num_edge_feats):
        if 'EGAT' in self.params.trainer:
            if isinstance(self.params.smiles, list):
                predictor = MultiCompEGATModel(self.params,num_node_feats,num_edge_feats)
            else:
                predictor = EGATModel(self.params,num_node_feats,num_edge_feats)
            
        elif 'NN' in self.params.trainer:
            if isinstance(self.params.smiles, list):
                predictor = MultiComponentFPModel(self.params)
            else:
                predictor = FPModel(self.params)
        elif 'LLM' in self.params.trainer:
            raise NotImplementedError("LLM model is not implemented yet.")
        elif 'EGAT+LLM' in self.params.trainer:
            raise NotImplementedError("GNN+LLM model is not implemented yet.")
        predictor = predictor.to(self.device)
        return predictor 

    def LoadAblationModel(self,predictorA,predictorB):
        if self.params.addons is not None:
            if self.params.graph == 'molecule':
                predictor = MolecularAblationModelwAddOns(self.params,predictorA,predictorB)
            elif self.params.graph == 'reaction':
                predictor = ReactionAblationModelwAddOns(self.params,predictorA,predictorB)
        else:
            if self.params.graph == 'molecule':
                predictor = MolecularAblationModel(self.params,predictorA,predictorB)
            elif self.params.graph == 'reaction':
                predictor = ReactionAblationModel(self.params,predictorA,predictorB)
        return predictor

    def ParallelizePredictor(self,predictor):
        if self.params.parallel:
            if torch.cuda.device_count() > 1:
                self.logger.info(f'Using {torch.cuda.device_count()} GPUs for parallel training.')
                predictor = nn.DataParallel(predictor)
            else:
                self.logger.info('Using single GPU for training.')
        return predictor

    def GetParams(self,predictor):
        predictor,total_params,trainable_params = FinetuningEGAT(self.params).freezeblocks(predictor)
        return predictor,total_params,trainable_params
    

    def grabnumfeats(self):
        firstsplit = self.dataloader.splits[0]
        datafunc = deepcopy(self.dataloader)
        datafunc()
        for split, loader in datafunc.egatdataloader.items():
            i = 0
            for stuff in loader:
                _ = stuff
                i += 1
                if i > 1: break
        self.node_feats = datafunc.egatdataset[firstsplit].node_feats_length
        self.edge_feats = datafunc.egatdataset[firstsplit].edge_feats_length

    
    def ablatorfunction(self,predictorA,predictorB):
        predictorA = self.LoadPredictor(self.node_feats,self.edge_feats)
        predictorB = self.LoadPredictor(self.node_feats,self.edge_feats)
        predictor = self.LoadAblationModel(predictorA,predictorB)
        return predictor

    def ObtainModelParams(self):
        if self.params.startpoint in ['EGAT_Ablation','NN_Ablation']:
            predictor = self.ablatorfunction(self.params.ablation_EGAT_model,self.params.ablation_NN_model)
        else:
            predictor = self.LoadPredictor(self.node_feats,self.edge_feats)
        
        predictor = self.ParallelizePredictor(predictor)
        predictor,total_params,trainable_params = self.GetParams(predictor)

        try:
            predictor.load_state_dict(self.checkpoint['model_state_dict'])
            print("Update predictor")
        except Exception as e:
            print(e)
            print('Cannot load State Dict. Starting Fresh Run.')
            #pass
        return predictor,total_params,trainable_params
    