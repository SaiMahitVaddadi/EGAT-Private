import torch.nn as nn
import torch
import dgl
from .blocks.aggblock import AggregationBlock
from .blocks.egatblock import EGATBlock
from .blocks.predictionblock import PredictionBlock


class FPModel(PredictionBlock):
    def __init__(self, cfg):
        super().__init__()
        self.params = cfg

        self.predblock = PredictionBlock(cfg)
    
    def forward(self, G_features,Hr=None):
        prediction = self.predblock.RunPrediction(G_features, Hr=Hr)
        return prediction

class MultiComponentFPModel(PredictionBlock):
    def __init__(self, cfg):
        super().__init__()
        self.params = cfg

        self.predblock = PredictionBlock(cfg)

    def forward(self, G_features,Hr=None):
        feats = [] 
        for i in range(len(G_features)):
            feats.append(G_features)
        G_features = torch.cat(feats, dim=1)
        prediction = self.predblock.RunPrediction(G_features, Hr=Hr)
        return prediction