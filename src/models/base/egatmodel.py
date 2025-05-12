import torch.nn as nn
import torch
import dgl
from .blocks.aggblock import AggregationBlock
from .blocks.egatblock import EGATBlock
from .blocks.predictionblock import PredictionBlock


class EGATModel(nn.Module):
    def __init__(self, cfg, num_node_feats=17, num_edge_feats=14,addonlen=0):
        super().__init__()
        self.params = cfg
        self.egatblock = EGATBlock(cfg, num_node_feats, num_edge_feats)
        if self.params.MixingLayer != False: 
            self.aggblock = AggregationBlock(cfg, addonlen=addonlen)
            self.predblock = PredictionBlock(cfg)
        else:
            self.aggblock = AggregationBlock(cfg)
            self.predblock = PredictionBlock(cfg, addonlen)
        
    
    def forward(self, graphR, graphP=None,Hr=None):
        Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_scores, P_attn_scores = self.egatblock.RunEGATBlock(graphR, graphP)
        G_features = self.aggblock.RunAggregationBlock(Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_scores, P_attn_scores)
        prediction = self.predblock.RunPrediction(G_features, Hr=Hr)
        return prediction
    
class MultiCompEGATModel(nn.Module):
    def __init__(self, cfg, num_node_feats=17, num_edge_feats=14):
        super().__init__()
        self.params = cfg

        self.egatblocks = [EGATBlock(cfg, num_node_feats, num_edge_feats) for _ in range(len(self.params.smiles) if self.params.smiles > 1 else 1) ]
        self.aggblock = [AggregationBlock(cfg) for _ in range(len(self.params.smiles) if self.params.smiles > 1 else 1)]
        self.componentaggblock = None
        self.predblock = PredictionBlock(cfg)

    def forward(self, graphR, graphP=None,Hr=None):
        feats = [] 
        for i in range(len(graphR)):
            Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_scores, P_attn_scores = self.egatblock[i].RunEGATBlock(graphR[i], graphP[i])
            G_features = self.aggblock[i].RunAggregationBlock(Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_scores, P_attn_scores)
            feats.append(G_features)
        G_features = torch.cat(feats, dim=1)
        prediction = self.predblock.RunPrediction(G_features, Hr=Hr)
        return prediction