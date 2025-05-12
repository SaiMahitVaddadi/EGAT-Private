import torch,logging,os,hydra
from torch import nn
from dataclasses import dataclass

@dataclass
class FinetuningParams:
    block: str = 'FreezeEGAT'
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    num_epochs: int = 10
class FinetuningEGAT:
    def __init__(self,arguments):
        self.params = arguments
        self.logger = logging.getLogger(__name__)

    def egatfreezing(self,predictor):
        self.egatfreeze = [predictor.egatblock.egat1,
                             predictor.egatblock.egat2]

    def aggfreezing(self,predictor):
        self.aggfreeze = []
        if hasattr(predictor.aggblock,'agg_N_feats'): 
            self.aggfreeze += [predictor.aggblock.agg_N_feats,
                               predictor.aggblock.agg_E_feats]
        if hasattr(predictor.aggblock,'aggfunc'):
            self.aggfreeze += [predictor.aggblock.aggfunc]
        
    def fpfreezing(self,predictor):
        self.fpfreeze = [predictor.predblock.mlp1,
                           predictor.predblock.mlp2,
                           predictor.predblock.mlp3]

    def freeze(self,layer):
        try:
            for param in layer.parameters():
                param.requires_grad = False
        except AttributeError:
            pass


    def freeze_layers(self,layers):
        for layer in layers:
            self.freeze(layer)


    def freezeblocks(self,predictor):
        self.egatfreezing(predictor)
        self.aggfreezing(predictor)
        self.fpfreezing(predictor)

        total_params = sum(p.numel() for p in predictor.parameters())

        if self.params.finetuner =='FreezeEGAT' or self.params.finetuner =='FinetuneNN':
            self.freeze_layers(self.egatfreeze)
            self.freeze_layers(self.aggfreeze)
        elif self.params.finetuner =='FreezeNN' or self.params.finetuner =='FinetuneEGAT':
            self.freeze_layers(self.fpfreeze)
        elif self.params.finetuner =='FreezeEGATOnly':
            self.freeze_layers(self.egatfreeze)
        elif self.params.finetuner =='FreezeAggOnly':
            self.freeze_layers(self.aggfreeze)
        elif self.params.finetuner =='FreezeNNOnly':
            self.freeze_layers(self.fpfreeze)
        elif self.params.finetuner =='FinetuneAgg':
            self.freeze_layers(self.egatfreeze)
            self.freeze_layers(self.fpfreeze)
        elif self.params.finetuner =='FreezeLastEGATLayer':
            self.freeze(predictor.egatblock.egat2)
            self.freeze_layers(self.aggfreeze)
            self.freeze_layers(self.fpfreeze)
        elif self.params.finetuner =='FreezeLastEGATLayerOnly':
            self.freeze(predictor.egatblock.egat2)
        elif self.params.finetuner =='FreezeLastEGATLayerwithAgg':
            self.freeze(predictor.egatblock.egat2)
            self.freeze_layers(self.aggfreeze)
        elif self.params.finetuner =='FreezeFirstEGATLayerwithAgg':
            self.freeze(predictor.egatblock.egat1)
            self.freeze_layers(self.aggfreeze)
        elif self.params.finetuner =='FreezeFirstEGATLayer':
            self.freeze(predictor.egatblock.egat1)
            self.freeze_layers(self.aggfreeze)
            self.freeze_layers(self.fpfreeze)
        elif self.params.finetuner =='FreezeFirstEGATLayerOnly':
            self.freeze(predictor.egatblock.egat1)
        
        trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        return predictor,total_params,trainable_params