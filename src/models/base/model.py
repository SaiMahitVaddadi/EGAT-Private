import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json
from ...layers.egat.dgl import EGATConvDGL,EGATConvResidDGL, EGATConvResidSADGL, EGATConvSADGL
from ..propertynet import PropertyNet
from dataclasses import dataclass
from typing import Optional, Union, List
from ..aggregators.dgl.attentive import AttentiveAggregator, CalcAttentiveAggregator
from ..aggregators.dgl.weighted import WeightedSumAggregator 
from ..aggregators.dgl.cluster import ClusterPooling
from ..aggregators.dgl.diffpool import DiffPool
from ..aggregators.dgl.edgepool import EdgePool,EdgePoolLayerwAttention
from ..aggregators.dgl.environment import EnvironmentAggregator,EnvironmentAggregatorwithGAT
from ..aggregators.dgl.sag import SAGPool,SAGPoolwAttention

'''
TO-DO:

- Fully custom activation functions
- Fully custom end convolution layers
- Fully custom softmax layers

'''
@dataclass
class ModelParams:
    model: str
    hidden_dim: int
    num_heads: int
    Resid: Optional[bool] = None
    ResidBias: Optional[bool] = None
    MessagePassing: Optional[bool] = False
    egatlayers: Optional[int] = 2
    Aggregate: Optional[str] = 'Concat'
    MixingLayer: Optional[bool] = False
    addons: Optional[bool] = None
    model_type: Optional[str] = 'default'
    targets: Optional[Union[str, List[str]]] = None
    dropout: Optional[float] = None
    NN_hidden_dim: Optional[Union[int, List[int]]] = 256
    cascading: Optional[bool] = False
    graph: Optional[str] = 'reaction'
    getattentionmaps: Optional[bool] = False
    getembeddings: Optional[int] = 0


class EGATModel(nn.Module):
    def __init__(self, cfg, num_node_feats = 17, num_edge_feats=14,addononlength=None,activation=None,endconvolution=None,softmax=None):
        super().__init__()
        self.params = cfg
        self.inputnodes = num_node_feats
        self.inputedges = num_edge_feats
        self.addonlength = addononlength
        self.activ = getattr(nn, activation) if activation is not None else None
        self.endconv = getattr(nn, endconvolution) if endconvolution is not None else None
        self.smax = nn.softmax(dim=1) if softmax is not None else None

        layer = self.SelectLayerDGL()
        self.InitializeEGAT(layer)
        self.InitializeAggregation()
        
        if '1MLP' in self.params.model:
            self.Initialize1MLP()
        elif '3MLP' in self.params.model:
            if 'Droupout' in self.params.model:
                if isinstance(self.params.targets, str):
                    self.Initialize3MLP1OUTDropout()
                else:
                    self.Initialize3MLPNOUTDropout()
            else:
                if isinstance(self.params.targets, str):
                    self.Initialize3MLP1OUT()
                else:
                    self.Initialize3MLPNOUT()
            

    def SelectLayerDGL(self):
        if self.params.Resid is not None: 
            if not self.params.SA:
                func = EGATConvResidDGL
            else:
                func = EGATConvResidSADGL
        else:
            if not self.params.SA:
                func = EGATConvDGL
            else:
                func = EGATConvSADGL
        
        return func 
    
    def InitializeEGAT(self,layer=EGATConvDGL):
        self.egat1 = layer(in_node_feats=self.inputnodes,in_edge_feats=self.inputedges,out_node_feats=self.params.hidden_dim,out_edge_feats=self.params.hidden_dim,num_heads=self.params.num_heads,edgeresid=self.params.Resid,bias=self.params.ResidBias)

        if self.params.MessagePassing:
            self.egat2 = layer(in_node_feats=self.params.hidden_dim*self.params.num_heads,in_edge_feats=self.params.hidden_dim*self.params.num_heads,out_node_feats=self.params.hidden_dim,out_edge_feats=self.params.hidden_dim,num_heads=self.params.num_heads,edgeresid=self.params.Resid,bias=self.params.ResidBias)
        else:
            self.egat2 = nn.ModuleList()
            # Intermediate GAT layers
            for _ in range(1, self.params.egatlayers):
                self.egat2.append(layer(in_node_feats=self.params.hidden_dim*self.params.num_heads,in_edge_feats=self.params.hidden_dim*self.params.num_heads,out_node_feats=self.params.hidden_dim,out_edge_feats=self.params.hidden_dim,num_heads=self.params.num_heads,edgeresid=self.params.Resid,bias=self.params.ResidBias))
        
    def InitializeAggregation(self):
        # BLOCK1: message-passing blocks
        if self.params.Aggregate == 'Concat':
            self.agg_N_feats = nn.Sequential(nn.Linear(2*self.params.hidden_dim*self.params.num_heads, 2*self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Linear(2*self.params.hidden_dim*self.params.num_heads, 2*self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
        elif self.params.Aggregate == 'Mixing':
            self.agg_N_feats = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads, self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads, self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
        else:
            # BLOCK2: aggregate reactant and product nodes features
            self.agg_N_feats = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads, self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads, self.params.hidden_dim*self.params.num_heads, bias=True), nn.GELU())

    def MixingLayer(self):  
        if self.params.MixingLayer:
            if self.params.Aggregate == 'Concat':
                self.Mixing_Layer = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads*2,self.params.hidden_dim*self.params.num_heads*2,self.params.hidden_dim*self.params.num_heads*4,bias=True),nn.GELU())
            else:
                self.Mixing_Layer = nn.Sequential(nn.Bilinear(self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads,self.params.hidden_dim*self.params.num_heads*2,bias=True),nn.GELU())
         
        if self.params.addons is not None:
            self.Mixing_Layer_RDkit = nn.Sequential(nn.Bilinear(self.hidden_dim*self.num_heads*2,self.addonlen,self.hidden_dim*self.num_heads*2,bias=True),nn.GELU())
        
    def Initialize1MLP(self):
        #BLOCK3: final MLP layers
        if self.params.model_type != 'BEP':
            if self.params.Aggregate == 'Concat':
                if not self.params.MixingLayer:
                    self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*4, 256, 1) for _ in range(len(self.params.target))])
                else:
                    self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*2, 256, 1) for _ in range(len(self.params.target))])
            else:
                self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*2, 256, 1) for _ in range(len(self.params.target))])
        else:
            if self.params.Aggregate == 'Concat':
                if not self.params.MixingLayer:
                    self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*4, 256, 1) for _ in range(2)])
                else:
                    self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*2, 256, 1) for _ in range(2)])
            else:
                self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*2, 256, 1) for _ in range(2)])

    def Initialize3MLP1OUT(self):
        if self.params.Aggregate == 'Concat':
            self.mlp1 = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads*4, 256, bias=True),nn.GELU())
        else:
            self.mlp1 = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads*2, 256, bias=True),nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(256, 128, bias=True),nn.GELU())
        self.mlp3 = nn.Linear(128, 1, bias=True)

    def Initialize3MLPNOUT(self):
        if self.params.Aggregate == 'Concat':
            self.mlp1 = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads*4, 256, bias=True),nn.GELU())
        else:
            self.mlp1 = nn.Sequential(nn.Linear(self.params.hidden_dim*self.params.num_heads*2, 256, bias=True),nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(256, 128, bias=True),nn.GELU())
        self.mlp3 = nn.Linear(128, len(self.params.targets), bias=True)
    
    def Initialize3MLP1OUTDropout(self):
        if self.params.Aggregate == 'Concat':
            layers = [nn.Linear(self.params.hidden_dim*self.params.num_heads*4, 256, bias=True), nn.GELU()]
            if self.params.dropout is not None:
                layers.append(nn.Dropout(p=self.params.dropout))
            self.mlp1 = nn.Sequential(*layers)
        else:
            layers = [nn.Linear(self.params.hidden_dim*self.params.num_heads*2, 256, bias=True), nn.GELU()]
            if self.params.dropout is not None:
                layers.append(nn.Dropout(p=self.params.dropout))
            self.mlp1 = nn.Sequential(*layers)
        
        layers = [nn.Linear(256, 128, bias=True), nn.GELU()]
        if self.params.dropout is not None:
            layers.append(nn.Dropout(p=self.params.dropout))
        self.mlp2 = nn.Sequential(*layers)
        
        self.mlp3 = nn.Linear(128, 1, bias=True)
    
    def Initialize3MLPNOUTDropout(self):
        if self.params.Aggregate == 'Concat':
            layers = [nn.Linear(self.params.hidden_dim*self.params.num_heads*4, 256, bias=True), nn.GELU()]
            if self.params.dropout is not None:
                layers.append(nn.Dropout(p=self.params.dropout))
            self.mlp1 = nn.Sequential(*layers)
        else:
            layers = [nn.Linear(self.params.hidden_dim*self.params.num_heads*2, 256, bias=True), nn.GELU()]
            if self.params.dropout is not None:
                layers.append(nn.Dropout(p=self.params.dropout))
            self.mlp1 = nn.Sequential(*layers)
        
        layers = [nn.Linear(256, 128, bias=True), nn.GELU()]
        if self.params.dropout is not None:
            layers.append(nn.Dropout(p=self.params.dropout))
        self.mlp2 = nn.Sequential(*layers)
        
        self.mlp3 = nn.Linear(128, len(self.params.targets), bias=True)

    def InitializeNMLP1OUT(self):
        if self.params.Aggregate == 'Concat':
            if isinstance(self.params.NN_hidden_dim, int):
                nn_hidden_dims = [self.params.NN_hidden_dim // (2 ** i) for i in range(3)] if self.params.cascading else [self.params.NN_hidden_dim] * 3
            else:
                nn_hidden_dims = self.params.NN_hidden_dim

            if self.params.Aggregate == 'Concat':
                layers = [nn.Linear(self.params.hidden_dim * self.params.num_heads * 4, nn_hidden_dims[0], bias=True), nn.GELU()]
                if self.params.dropout is not None:
                    layers.append(nn.Dropout(p=self.params.dropout))
                self.mlp1 = nn.Sequential(*layers)
            else:
                layers = [nn.Linear(self.params.hidden_dim * self.params.num_heads * 2, nn_hidden_dims[0], bias=True), nn.GELU()]
                if self.params.dropout is not None:
                    layers.append(nn.Dropout(p=self.params.dropout))
                self.mlp1 = nn.Sequential(*layers)

            layers = [nn.Linear(nn_hidden_dims[0], nn_hidden_dims[1], bias=True), nn.GELU()]
            if self.params.dropout is not None:
                layers.append(nn.Dropout(p=self.params.dropout))
            self.mlp2 = nn.Sequential(*layers)

            self.mlp3 = nn.Linear(nn_hidden_dims[1], 1, bias=True)

    def InitializeNMLPNOUT(self):
        if self.params.Aggregate == 'Concat':
            if isinstance(self.params.NN_hidden_dim, int):
                nn_hidden_dims = [self.params.NN_hidden_dim // (2 ** i) for i in range(3)] if self.params.cascading else [self.params.NN_hidden_dim] * 3
            else:
                nn_hidden_dims = self.params.NN_hidden_dim

            if self.params.Aggregate == 'Concat':
                layers = [nn.Linear(self.params.hidden_dim * self.params.num_heads * 4, nn_hidden_dims[0], bias=True), nn.GELU()]
                if self.params.dropout is not None:
                    layers.append(nn.Dropout(p=self.params.dropout))
                self.mlp1 = nn.Sequential(*layers)
            else:
                layers = [nn.Linear(self.params.hidden_dim * self.params.num_heads * 2, nn_hidden_dims[0], bias=True), nn.GELU()]
                if self.params.dropout is not None:
                    layers.append(nn.Dropout(p=self.params.dropout))
                self.mlp1 = nn.Sequential(*layers)

            layers = [nn.Linear(nn_hidden_dims[0], nn_hidden_dims[1], bias=True), nn.GELU()]
            if self.params.dropout is not None:
                layers.append(nn.Dropout(p=self.params.dropout))
            self.mlp2 = nn.Sequential(*layers)

            self.mlp3 = nn.Linear(nn_hidden_dims[1], len(self.params.targets), bias=True)

    def InitializeNMLPSubnets(self):
        #BLOCK3: final MLP layers
        if self.params.model_type != 'BEP':
            if self.params.Aggregate == 'Concat':
                if not self.params.MixingLayer:
                    self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*4, 256, 1) for _ in range(len(self.params.target))])
                else:
                    self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*2, 256, 1) for _ in range(len(self.params.target))])
            else:
                self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*2, 256, 1) for _ in range(len(self.params.target))])
        else:
            if self.params.Aggregate == 'Concat':
                if not self.params.MixingLayer:
                    self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*4, 256, 1) for _ in range(2)])
                else:
                    self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*2, 256, 1) for _ in range(2)])
            else:
                self.subnets = nn.ModuleList([PropertyNet(self.params.hidden_dim*self.params.num_heads*2, 256, 1) for _ in range(2)])
        pass

    def InitializeAggregatorFunctions(self):
        if self.params.pooling == 'WtSum':
            self.agg_func = WeightedSumAggregator(self.inputnodes,self.inputedges)
        elif self.params.pooling == 'LearnedAttn':
            self.agg_func = AttentiveAggregator(self.inputnodes,self.inputedges)
        elif self.params.pooling == 'CalcedAttn':
            self.agg_func = CalcAttentiveAggregator(self.inputnodes,self.inputedges)
        

    def LayerOne(self, graphR, graphP=None):
        Rnode_feats, Redge_feats = self.egat1(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.params.hidden_dim * self.params.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.params.hidden_dim * self.params.num_heads)

        if self.params.graph == 'reaction':
            Pnode_feats, Pedge_feats = self.egat1(graphP, graphP.ndata['x'], graphP.edata['x'])
            Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.params.hidden_dim * self.params.num_heads)
            Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.params.hidden_dim * self.params.num_heads)
        else:
            Pnode_feats, Pedge_feats = None, None

        return Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats
    

    def LayerTwoNoMP(self, graphR, graphP=None):
        Rnode_feats, Redge_feats = self.egat2(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.params.hidden_dim * self.params.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.params.hidden_dim * self.params.num_heads)
        if self.params.graph == 'reaction':
            Pnode_feats, Pedge_feats = self.egat2(graphP, graphP.ndata['x'], graphP.edata['x'])
            Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.params.hidden_dim * self.params.num_heads)
            Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.params.hidden_dim * self.params.num_heads)
        else:
            Pnode_feats, Pedge_feats = None, None

        return Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats

    def GetAttentionMaps(self,graph,i):
        if self.getattentionmaps and i == self.egatlayers:
            R_attn_scores = self.egat2.edge_attn
            if self.params.num_heads > 1:R_attn_scores = torch.norm(R_attn_scores,dim=1)
            graph.edata['norm_attn'] = R_attn_scores
            if self.SA:
                R_self_attn = self.egat2.self_attn
                if self.params.num_heads > 1:R_self_attn = torch.norm(R_self_attn,dim=1)
                graph.ndata['norm_attn'] = R_self_attn
        
            # Initialize a square matrix with zeros
            matrix_size = graph.number_of_nodes()
            R_combined_matrix = torch.zeros(matrix_size, matrix_size)
            # Fill the off-diagonal with edge attention scores
            src, dst = graph.edges()
            R_combined_matrix[src, dst] = graph.edata['norm_attn'].view(-1)

            if self.SA:
                # Fill the diagonal with node self-attention scores
                R_combined_matrix.fill_diagonal_(graph.ndata['norm_attn'].view(-1))
            return R_combined_matrix
        else:
            return None


    def LayerTwoMP(self, graphR, graphP=None):
        for i in range(self.egatlayers-1):
                Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
                Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.params.hidden_dim * self.params.num_heads)
                Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.params.hidden_dim * self.params.num_heads)
                
                R_combined_matrix = self.GetAttentionMaps(graphR,i)
                if self.params.graph == 'reaction':
                    Pnode_feats, Pedge_feats = self.egat2(graphP, Pnode_feats, Pedge_feats)
                    Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.params.hidden_dim * self.params.num_heads)
                    Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.params.hidden_dim * self.params.num_heads)
                    P_combined_matrix = self.GetAttentionMaps(graphP,i)
                else:
                    Pnode_feats, Pedge_feats,P_combined_matrix = None, None,None

        return Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats,R_combined_matrix,P_combined_matrix


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
    
    
              
    

    def AttentiveAggv2(self,individual_graphs,node_feats=17,edge_feats=14):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
            # Compute attention scores (logits)
            node_logits = self.node_attn(node_feats).squeeze()  # [N]
            node_e = node_logits.exp()                          # [N]
            node_z = node_e.sum()                               # Scalar (sum for normalization)
            node_alphas = node_e / node_z                       # [N]
            # Weighted sum of node features
            global_node_feature = (node_alphas.unsqueeze(-1) * node_feats).sum(dim=0)  # [d_node]

            # --- Edge Attention ---
            edge_feats = graph.edata['x']
            # Compute attention scores (logits)
            edge_logits = self.edge_attn(edge_feats).squeeze()  # [E]
            edge_e = edge_logits.exp()                          # [E]
            edge_z = edge_e.sum()                               # Scalar
            edge_alphas = edge_e / edge_z                       # [E]
            # Weighted sum of edge features
            global_edge_feature = (edge_alphas.unsqueeze(-1) * edge_feats).sum(dim=0)  # [d_edge]

            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)
        
        return torch.stack(G_node_feats), torch.stack(G_edge_feats)

            
    def BondAgg(self,individual_graphs):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
            # Weighted node aggregation based on bond features
            bond_feats = graph.edata['x']
            bond_weights = torch.sum(bond_feats, dim=1)



    def Aggregation(self,Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR):
        if self.aggregate == 'Concat':
            Rxn_node_feature = self.agg_N_feats(torch.cat(Pnode_feats,Rnode_feats))
            Rxn_edge_feature = self.agg_E_feats(torch.cat(Pedge_feats,Redge_feats))
        else:
            Rxn_node_feature = self.agg_N_feats(Pnode_feats - Rnode_feats)
            Rxn_edge_feature = self.agg_E_feats(Pedge_feats - Redge_feats)

        # obtain global feature for larer 3
        graphR.ndata['x'] = Rxn_node_feature
        graphR.edata['x'] = Rxn_edge_feature 
        individual_graphs = dgl.unbatch(graphR)

        if self.params.pooling == 'Sum':
            G_node_feats,G_edge_feats = self.SumAgg(individual_graphs)
        elif self.params.pooling == 'Norm':
            G_node_feats,G_edge_feats = self.NormAgg(individual_graphs)
        elif self.params.pooling == 'Mean':
            G_node_feats,G_edge_feats = self.MeanAgg(individual_graphs)
        elif self.params.pooling == 'Std':
            G_node_feats,G_edge_feats = self.StdAgg(individual_graphs)
        
        if self.MixingLayer:
            G_features = self.Mixing_Layer(G_node_feats,G_edge_feats)
            
        else:
            G_features = torch.cat((G_node_feats,G_edge_feats), axis=1)
        return G_features
    
    def OneMLP(self,G_features):
        x = [subnet(G_features) for subnet in self.subnets]
        x = torch.cat(x, dim=1)
        return x
    
    def ThreeMLP(self,G_features,Hr=None):
        x   = self.mlp1(G_features)
        x   = self.mlp2(x)

        # merge features with Hr, followed by MLP
        if 'Hr' in self.params.model_type and 'Hr2' not in self.params.model_type: x = torch.cat((x,Hr), axis=1)
        x   = self.mlp3(x)
        return x


    def forward(self, graphR, graphP,Hr=None):
        ##################################### 
        ############# layer one ############# 
        ##################################### 
        Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats = self.LayerOne(graphR, graphP)

        ##################################### 
        ############# layer two ############# 
        ##################################### 
        if self.MessagePassing:
            Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats,R_combined_matrix,P_combined_matrix = self.LayerTwoMP(graphR, graphP)
        else:
            Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats = self.LayerTwoNoMP(graphR, graphP)
            R_combined_matrix,P_combined_matrix = None,None

        G_features = self.Aggregation(Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR)
            
        #################################### 
        ########## merge features ##########
        #################################### 
        # MLP
        if 'Hr2' in self.params.model_type: G_features= torch.cat((G_features,Hr),axis=1)
        if '1MLP' in self.params.model: 
            x = self.OneMLP(G_features)
        elif '3MLP' in self.params.model:
            x = self.ThreeMLP(G_features,Hr)

        if self.smax == 'smax':
            x = self.smax(x)
            
        
        if self.getembeddings == 0:
            if self.getattentionmaps:
                return x,R_combined_matrix,P_combined_matrix
            else:
                return x
        elif self.getembeddings == 1:
            if self.getattentionmaps:
                return x,G_features,R_combined_matrix,P_combined_matrix
            else:
                return x,G_features
        elif self.getembeddings == 2:
            if self.getattentionmaps:
                return G_features,R_combined_matrix,P_combined_matrix
            else:
                return G_features
