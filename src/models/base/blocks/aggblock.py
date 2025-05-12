import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json
from ....models.aggregators.dgl.attentive import AttentiveAggregator,CalcAttentiveAggregator,CalcAttentiveAggregatorMLP
from ....models.aggregators.dgl.bondagg import BondAggregator,BondEnvironmentAggregator,BondEnvironmentAggregatorSimplified
from ....models.aggregators.dgl.weighted import WeightedSumAggregator,LearnedPooledAggregatorwithWeightedFeatures,WeightedMeanAggregator,LearnedPooledAggregator,BasicPooledAggregator
from ....models.aggregators.dgl.cluster import ClusterPooling
from ....models.aggregators.dgl.diffpool import DiffPool
from ....models.aggregators.dgl.edgepool import EdgePool,EdgePoolwAttention
from ....models.aggregators.dgl.environment import EnvironmentAggregator,EnvironmentAggregatorwithGAT
from ....models.aggregators.dgl.pathintegral import PIPooling
from ....models.aggregators.dgl.sag import SAGPool,SAGPoolwAttention
from dataclasses import dataclass
from typing import Optional, Union
#You can add more aggregators here as needed

@dataclass
class AggregationBlockParams:
    hidden_dim: int
    num_heads: int
    AggregateReaction: str = 'Concat'  # Options: 'Concat', 'Mixing', etc.
    Aggregate: str = 'Concat'  # Options: 'Concat', etc.
    MixingLayer: Union[str, bool] = 'Bilinear'  # Options: 'Bilinear', 'Linear', True, False
    pooling: str = 'Sum'  # Options: 'WtSum', 'LearnedAttn', 'CalcedAttn', etc.
    graph: str = 'reaction'  # Options: 'reaction', 'molecule'
    addons: Optional[list] = None  # Additional features, if any
    addonlen: Optional[int] = None  # Length of additional features
    norm: float = 1.0  # Normalization factor


class BasicAggregator(nn.Module):
    def __init__(self, cfg, ):
        super().__init__()
        self.params = cfg

    def SumAgg(self,individual_graphs):
        G_node_feats,G_edge_feats = [],[]
        for graph in individual_graphs:
            global_node_feature = graph.ndata['x'].sum(dim=0)
            global_edge_feature = graph.edata['x'].sum(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)
        return G_node_feats,G_edge_feats
    
    def NormAgg(self,individual_graphs):
        G_node_feats,G_edge_feats = [],[]
        for graph in individual_graphs:
            global_node_feature = graph.ndata['x'].sum(dim=0)
            global_edge_feature = graph.edata['x'].sum(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

        G_node_feats = G_node_feats/self.params.norm
        G_edge_feats = G_edge_feats/self.params.norm

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)
        return G_node_feats,G_edge_feats

    def MeanAgg(self,individual_graphs):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
            global_node_feature = graph.ndata['x'].mean(dim=0)
            global_edge_feature = graph.edata['x'].mean(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)
        return G_node_feats, G_edge_feats

    def StdAgg(self,individual_graphs):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
            global_node_feature = graph.ndata['x'].std(dim=0)
            global_edge_feature = graph.edata['x'].std(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)
        return G_node_feats, G_edge_feats
    
    def NodeandEdgeAggFcn(self,individual_graphs,aggfcn):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
            global_node_feature,global_edge_feature = aggfcn(graph)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)
        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)
        return G_node_feats, G_edge_feats
    
    def FeaturesAggFcn(self,individual_graphs,aggfcn):
        G_features = []
        for graph in individual_graphs:
            global_feature = aggfcn(graph)
            G_features.append(global_feature)
        G_features = torch.stack(G_features)
        return G_features



class MixingLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.params = cfg
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.setupfunction()

    def setupfunction(self):  
        if self.params.MixingLayer == 'Bilinear' or self.params.MixingLayer == True:
            self.bilinearsetup()
        elif self.params.MixingLayer == 'Linear':
            self.linearsetup()

    def bilinearsetup(self):
        if self.params.Aggregate == 'Concat':
            self.Mixing_Layer = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads*2,self.params.hidden_dim*self.params.num_heads*2,self.params.hidden_dim*self.params.num_heads*4,bias=True),nn.GELU())
        else:
            self.Mixing_Layer = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads*2,bias=True),nn.GELU())
     
    def linearsetup(self):
        if self.params.Aggregate == 'Concat':
            self.Mixing_Layer = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads*2,self.params.hidden_dim*self.params.num_heads*4,bias=True),nn.GELU())
        else:
            self.Mixing_Layer = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads*2,bias=True),nn.GELU())
   
    def forward(self, G_node_feats, G_edge_feats):
        if self.params.MixingLayer != False:
            G_features = self.mixinglayer(G_node_feats, G_edge_feats)
        else:
            G_features = torch.cat((G_node_feats, G_edge_feats), axis=1)
        return G_features


class AddonMixingLayer(nn.Module):
    def __init__(self, cfg, addonlen):
        super().__init__()
        self.params = cfg
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.addonlen = addonlen
        self.setupfunc()

    def setupfunc(self):
        if self.params.addons is not None and self.params.addonmixing != False:
            if self.params.addonmixing == 'bilinear':
                self.Mixing_Layer_RDkit = nn.Sequential(nn.Bilinear(self.hidden_dim*self.num_heads*2,self.addonlen,self.hidden_dim*self.num_heads*2,bias=True),nn.GELU())
            else:
                self.Mixing_Layer_RDkit = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads*2+self.addonlen,self.hidden_dim*self.num_heads*2,bias=True),nn.GELU())

    def forward(self, G_node_feats, addonvector=None):
        if self.params.addons is not None and self.params.addonmixing != False:
            if self.params.addonmixing == 'bilinear':
                G_features = self.Mixing_Layer_RDkit(G_node_feats, addonvector)
            else:
                G_node_feats = torch.cat((G_node_feats, addonvector), axis=1)
                G_features = self.Mixing_Layer_RDkit(G_node_feats)
        else:
            G_features = G_node_feats
        return G_features


class AggregationBlock(BasicAggregator):
    def __init__(self, cfg,addonlen=None):
        super().__init__(cfg)
        self.params = cfg
        self.addonlen = addonlen
        self.SetupAggregationBlock()
    
    def NodeEdgeAggregationLayers(self):
        # BLOCK1: message-passing blocks
        if self.params.AggregateReaction == 'Concat':
            self.agg_N_feats = nn.Sequential(nn.Linear(2*self.params.hidden_dim*self.params.num_heads, 2*self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Linear(2*self.params.hidden_dim*self.params.num_heads, 2*self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
        elif self.params.AggregateReaction == 'Mixing':
            self.agg_N_feats = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads, self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads, self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
        else:
            # BLOCK2: aggregate reactant and product nodes features
            self.agg_N_feats = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads, self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads, self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())

    def bilinearmixinglayer(self):
        if self.params.Aggregate == 'Concat':
            self.Mixing_Layer = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads*2,self.params.hidden_dim*self.params.num_heads*2,self.params.hidden_dim*self.params.num_heads*4,bias=True),nn.GELU())
        else:
            self.Mixing_Layer = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads*2,bias=True),nn.GELU())
        
    def MixingLayer(self):  
        self.mixinglayer = MixingLayer(self.params)
        self.addonmixinglayer = AddonMixingLayer(self.params, self.addonlen)

    def setinputnodesandedges(self):
        if self.params.AggregateReaction == 'Concat':
            self.inputnodes = 2*self.params.hidden_dim*self.params.num_heads
            self.inputedges = 2*self.params.hidden_dim*self.params.num_heads
        else:
            self.inputnodes = self.params.hidden_dim*self.params.num_heads
            self.inputedges = self.params.hidden_dim*self.params.num_heads

    def AggregatorFunctions(self):
        self.setinputnodesandedges()
        if self.params.pooling == 'WtSum':
            self.agg_func = WeightedSumAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'LearnedAttn':
            self.agg_func = AttentiveAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'CalcedAttn':
            self.agg_func = CalcAttentiveAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'CalcedAttnMLP':
            self.agg_func = CalcAttentiveAggregatorMLP(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'BondAgg':
            self.agg_func = BondAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'BondEnvAgg':
            self.agg_func = BondEnvironmentAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'BondEnvAggSimplified':
            self.agg_func = BondEnvironmentAggregatorSimplified(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'WeightedMean':
            self.agg_func = WeightedMeanAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'LearnedPooled':
            self.agg_func = LearnedPooledAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'LearnedPooledWeighted':
            self.agg_func = LearnedPooledAggregatorwithWeightedFeatures(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'BasicPooled':
            self.agg_func = BasicPooledAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'Cluster':
            self.agg_func = ClusterPooling(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'DiffPool':
            self.agg_func = DiffPool(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'EdgePool':
            self.agg_func = EdgePool(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'EdgePoolAttn':
            self.agg_func = EdgePoolwAttention(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'Environment':
            self.agg_func = EnvironmentAggregator(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'EnvironmentGAT':
            self.agg_func = EnvironmentAggregatorwithGAT(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'PIPooling':
            self.agg_func = PIPooling(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'SAGPool':
            self.agg_func = SAGPool(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'SAGPoolAttn':
            self.agg_func = SAGPoolwAttention(self.inputnodes, self.inputedges)
        elif self.params.pooling == 'Sum':
            self.agg_func = self.SumAgg
        elif self.params.pooling == 'Norm':
            self.agg_func = self.NormAgg
        elif self.params.pooling == 'Mean':
            self.agg_func = self.MeanAgg
        elif self.params.pooling == 'Std':
            self.agg_func = self.StdAgg

        self.outputsnodeandedgepooledvectors = [self.SumAgg,self.NormAgg,self.MeanAgg,self.StdAgg]
        self.externaloutputsnodeandedgepooledvectors = [CalcAttentiveAggregator,CalcAttentiveAggregatorMLP,
                                                AttentiveAggregator,WeightedSumAggregator,WeightedMeanAggregator,
                                                LearnedPooledAggregator,BasicPooledAggregator]
        self.outputsglobalvectors = [] 

    def getreactionnodefeatures(self,Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR):
        if self.aggregate == 'Concat':
            Rxn_node_feature = self.agg_N_feats(torch.cat(Pnode_feats,Rnode_feats))
            Rxn_edge_feature = self.agg_E_feats(torch.cat(Pedge_feats,Redge_feats))
        else:
            Rxn_node_feature = self.agg_N_feats(Pnode_feats - Rnode_feats)
            Rxn_edge_feature = self.agg_E_feats(Pedge_feats - Redge_feats)
        return Rxn_node_feature,Rxn_edge_feature

    def createreactiongraph(self,Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR):
        Rxn_node_feature,Rxn_edge_feature = self.getreactionnodefeatures(Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR)
        # obtain global feature for larer 3
        graphR.ndata['x'] = Rxn_node_feature
        graphR.edata['x'] = Rxn_edge_feature 
        individual_graphs = dgl.unbatch(graphR)
        return individual_graphs

    def createmoleculegraph(self,Rnode_feats,Redge_feats,graphR):
        # merge R and P features
        Rxn_node_feature = self.agg_N_feats(Rnode_feats)
        Rxn_edge_feature = self.agg_E_feats(Redge_feats)
        # obtain global feature for layer 3
        graphR.ndata['x'] = Rxn_node_feature
        graphR.edata['x'] = Rxn_edge_feature 
        individual_graphs = dgl.unbatch(graphR)
        return individual_graphs
        
    def obtainindivgraphs(self,Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR):
        if self.params.graph == 'reaction':
            individual_graphs = self.createreactiongraph(Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR)
        else:
            individual_graphs = self.createmoleculegraph(Rnode_feats,Redge_feats,graphR)
        return individual_graphs
    
    def usemixinglayer(self,G_node_feats,G_edge_feats,addonvector=None):
        G_features = self.mixinglayer(G_node_feats,G_edge_feats)
        if self.params.addons is not None:
            G_features = self.addonmixinglayer(G_features,addonvector)
        else:
            G_features = G_features
        return G_features
    
    def usepoolingfcn(self,individual_graphs):
        if self.aggfcn in self.outputsnodeandedgepooledvectors:
            G_node_feats,G_edge_feats = self.agg_func(individual_graphs)
            G_features = self.usemixinglayer(G_node_feats,G_edge_feats)
        elif self.aggfcn in self.externaloutputsnodeandedgepooledvectors:
            G_node_feats,G_edge_feats = self.NodeandEdgeAggFcn(individual_graphs,self.agg_func)
            G_features = self.usemixinglayer(G_node_feats,G_edge_feats)
        else:
            G_features = self.FeaturesAggFcn(individual_graphs,self.agg_func)
        return G_features
    
    def Aggregation(self,Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR):
        individual_graphs = self.obtainindivgraphs(Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR)
        G_features = self.usepoolingfcn(individual_graphs)
        return G_features
    
    def SetupAggregationBlock(self):
        self.NodeEdgeAggregationLayers()
        self.AggregatorFunctions()
        self.MixingLayer()
        self.setinputnodesandedges()

    def RunAggregationBlock(self,Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR):
        G_features = self.Aggregation(Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR)
        return G_features
    