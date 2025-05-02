import torch,logging,os,hydra
from torch import nn


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


    def freezeblocks(self,predictor,block='FreezeEGAT'):
        self.egatfreezing(predictor)
        self.aggfreezing(predictor)
        self.fpfreezing(predictor)

        total_params = sum(p.numel() for p in predictor.parameters())

        if block == 'FreezeEGAT' or block == 'FinetuneNN':
            self.freeze_layers(self.egatfreeze)
            self.freeze_layers(self.aggfreeze)
        elif block == 'FreezeNN' or block == 'FinetuneEGAT':
            self.freeze_layers(self.fpfreeze)
        elif block == 'FreezeEGATOnly':
            self.freeze_layers(self.egatfreeze)
        elif block == 'FreezeAggOnly':
            self.freeze_layers(self.aggfreeze)
        elif block == 'FreezeNNOnly':
            self.freeze_layers(self.fpfreeze)
        elif block == 'FinetuneAgg':
            self.freeze_layers(self.egatfreeze)
            self.freeze_layers(self.fpfreeze)
        elif block == 'FreezeLastEGATLayer':
            self.freeze(predictor.egatblock.egat2)
            self.freeze_layers(self.aggfreeze)
            self.freeze_layers(self.fpfreeze)
        elif block == 'FreezeLastEGATLayerOnly':
            self.freeze(predictor.egatblock.egat2)
        elif block == 'FreezeLastEGATLayerwithAgg':
            self.freeze(predictor.egatblock.egat2)
            self.freeze_layers(self.aggfreeze)
        elif block == 'FreezeFirstEGATLayerwithAgg':
            self.freeze(predictor.egatblock.egat1)
            self.freeze_layers(self.aggfreeze)
        elif block == 'FreezeFirstEGATLayer':
            self.freeze(predictor.egatblock.egat1)
            self.freeze_layers(self.aggfreeze)
            self.freeze_layers(self.fpfreeze)
        elif block == 'FreezeFirstEGATLayerOnly':
            self.freeze(predictor.egatblock.egat1)
        
        trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        return predictor,total_params,trainable_params