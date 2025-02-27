import torch,logging,os,sys,hydra,omegaconf,shutil,importlib
from torch import nn
from dataclasses import dataclass
from typing import Optional

from ..models.model import EGATModel
from ..models.fpmodel import FPModel
from ..models.ablation.ablationmodels import MolecularAblationModel,ReactionAblationModel,MolecularAblationModelwAddOns,ReactionAblationModelwAddOns



@dataclass
class SetupParams:
    data_path: str
    save_path: str
    model_path: Optional[str] = None
    base_model: Optional[str] = None
    ablation_EGAT_model: Optional[str] = None
    ablation_NN_model: Optional[str] = None
    loss: str = 'CrossEntropy'
    metric: str = 'Accuracy'
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    learning_rate_min: float = 1e-5
    momentum_orig: float = 0.9
    lr_decay: float = 0.1
    step_size: int = 10
    weight_decay: float = 1e-4
    epochs: int = 100
    gpu: int = 0
    parallel: bool = False
    setup: str = 'cuda'
    startpoint: str = 'Retrain'
    weightsandbiases: bool = False
    wandbproject: Optional[str] = None
    wandbname: Optional[str] = None
    model: str = 'EGAT'
    hasaddons: bool = False
    molecular: bool = False



def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum
    
    return m,momentum


class Setup:
    def __init__(self,arguments):
        self.params = arguments
        ### Set Logger
        self.logger = logging.getLogger(__name__)
        self.LoadWandB()
        self.device = self.LoadTorchSetup()
        self.loss = self.LoadLoss()

        self.best_loss,self.start_epoch,self.checkpoint = self.LoadModel()
        self.model,self.total_params,self.trainable_params = self.ObtainModelParams()
        self.optimizer = self.LoadOptimizer(self.model)
        self.scheduler = self.LoadScheduler(self.optimizer)
        self.metric = self.LoadMetric()
        self.global_epoch = 0
        self.loss_increase_count = 0

    

    def LoadWandB(self):
        ### Check if weights and biases is needed for live model monitoring
        if self.params.weightsandbiases:
            import wandb

            wandb.init(project=self.params.wandbproject,name=self.params.wandbname)
    
    def LoadTorchSetup(self):
        ### Get the gradient enabled
        torch.set_grad_enabled(True)

        ### Set the GPU if available. Otherwise Stop. 
        if torch.cuda.is_available() and self.params.setup == 'cuda':
            if torch.cuda.device_count() > 1 and self.params.parallel:
                device = torch.device("cuda")
                self.logger.info(f'Using {torch.cuda.device_count()} GPUs for parallel training.')
            else:
                device = torch.device("cuda:0")
                self.logger.info('Using single GPU for training.')
        else:
            if torch.backends.mps.is_available() and self.params.setup == 'mps':
                if torch.cuda.device_count() > 1 and self.params.parallel:
                    device = torch.device("mps")
                    self.logger.info(f'Using {torch.cuda.device_count()} MPS devices for parallel training.')
                else:
                    device = torch.device("mps")
                    self.logger.info('Using single MPS device for training.')
            elif torch.backends.hip.is_available() and self.params.setup == 'rocm':
                if torch.cuda.device_count() > 1 and self.params.parallel:
                    device = torch.device("hip")
                    self.logger.info(f'Using {torch.cuda.device_count()} ROCm devices for parallel training.')
                else:
                    device = torch.device("hip")
                    self.logger.info('Using single ROCm device for training.')
            else:
                self.logger.info('GPU does not load. Only Run with CPU.')
                device = torch.device("cpu")
        return device
    
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

    def LoadModel(self):
        if self.params.startpoint == 'Retrain': 
            try:
                best_loss, start_epoch,checkpoint = self.PreLoadModel()
            except:
                best_loss, start_epoch,checkpoint = self.StartFromScratch()

            return best_loss, start_epoch,checkpoint
        ### Attempt to find a pre-trained model. If not, stop.
        elif self.params.startpoint in ['Pretrain','EGAT_Freeze','NN_Freeze','EGAT_Finetune','NN_Finetune']:                
            try:
                best_loss, start_epoch,checkpoint = self.LoadModelForPretraining(self.params.base_model)
            except:
                best_loss, start_epoch,checkpoint = self.StartFromScratch()
            return best_loss, start_epoch,checkpoint
        ### Attempt to find a pre-trained model and freeze the EGAT side
        elif self.params.startpoint == 'EGAT_Ablation' or self.params.startpoint == 'NN_Ablation': # Attempt to find two pre-trained models. If not, stop.
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
        elif self.params.startpoint == 'Predict':
            try:
                best_loss, start_epoch,checkpoint = self.PreLoadModel(self.params.model_path)
            except:
                best_loss, start_epoch,checkpoint = self.StartFromScratch()
            return best_loss, start_epoch,checkpoint
        # Attempt to start from scratch.
        else: 
            best_loss, start_epoch,checkpoint = self.StartFromScratch()
            return best_loss, start_epoch,checkpoint


    def LossLoader(self,lossfcn):
        try:
            loss = getattr(torch,nn,f"{self.params.loss}Loss")
            loss_fn = loss()
        except:
            loss_fn = None
            raise ValueError(f'Loss function not {self.params.loss} supported')
        return loss_fn

    def LoadLoss(self):
        ### Load the loss function
        if isinstance(self.params.loss,str):
            loss_fn = self.LossLoader(self.params.loss)
        elif isinstance(self.params.loss,list):
            loss_fn = []
            for loss in self.params.loss:
                loss_fn.append(self.LossLoader(loss))
        return loss_fn
    
    def LoadMetric(self):
        ### Load the loss function
        if isinstance(self.params.metric,str):
            loss_fn = self.LossLoader(self.params.metric)
        elif isinstance(self.params.metric,list):
            loss_fn = []
            for loss in self.params.metric:
                loss_fn.append(self.LossLoader(loss))
        return loss_fn
        
        
    def InitializeModel(self):
        '''INITALIIZATION of DATA'''
        ### Get the data path, otherwise stop if it's not found. 
        try:
            root = hydra.utils.to_absolute_path(self.params.data_path) # Get the data path
            return True
        except:
            self.logger.info('Data Path cannot be found')
            return None    

    def LoadPredictor(self):
        if 'EGAT' in self.params.model:
            predictor = EGATModel(self.params).to(self.device)
        elif 'NN' in self.params.model:
            predictor = FPModel(self.params).to(self.device)
        return predictor 

    def LoadAblationModel(self,predictorA,predictorB):
        if self.params.hasaddons:
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
        total_params = sum(p.numel() for p in predictor.parameters())

        freeze_layers = dict()
        freeze_layers['EGAT_Freeze'] = [predictor.egat1,predictor.egat2,predictor.agg_N_feats,predictor.agg_E_feats]
        freeze_layers['NN_Freeze'] = [predictor.mlp1,predictor.mlp2,predictor.mlp3]


        freeze_layers_nomp = dict()
        freeze_layers_nomp_1mlp = dict()
        freeze_layers_1mlp = dict()
        
        if self.params.startpoint in ['EGAT_Freeze','NN_Freeze']:
            # Freeze the parameters of egats
            layers = freeze_layers[self.params.startpoint]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        return predictor,total_params,trainable_params

    def ObtainModelParams(self):
        if self.params.startpoint in ['EGAT_Ablation','NN_Ablation']:
            predictorA = self.LoadPredictor(self.params.ablation_EGAT_model)
            predictorB = self.LoadPredictor(self.params.ablation_NN_model)
            predictor = self.LoadAblationModel(predictorA,predictorB)
        else:
            predictor = self.LoadPredictor(self.params.defaults.model)
        
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
    


    def LoadScheduler(self,optimizer):

        self.LEARNING_RATE_CLIP = self.params.learning_rate_min
        self.MOMENTUM_ORIGINAL = self.params.momentum_orig #.1
        self.MOMENTUM_DECAY = self.params.lr_decay
        self.MOMENTUM_DECAY_STEP = self.params.step_size

        ### Load the scheduler
        if self.params.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.epochs, eta_min=0)
        elif self.params.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.params.step_size, gamma=self.params.gamma)
        elif self.params.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return scheduler
    
    def LoadOptimizer(self,model,lr=None):
        ### Load the optimizer
        if lr is None: lr = self.params.lr
        if self.params.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
        elif self.params.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
        elif self.params.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
        else:
            raise ValueError('Optimizer not supported')
        return optimizer

    