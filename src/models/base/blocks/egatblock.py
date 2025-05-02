import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json
from ...layers.egat.dgl import EGATConvDGL,EGATConvResidDGL, EGATConvResidSADGL, EGATConvSADGL


class EGATBlock(nn.Module):
    def __init__(self, cfg, num_node_feats = 17, num_edge_feats=14):
        super().__init__()
        self.params = cfg
        self.inputnodes = num_node_feats
        self.inputedges = num_edge_feats
        self.SelectLayer()
        self.InitializeEGAT()

    def selectlayerDGL(self):
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
        self.layer = func
    
    def SelectLayer(self):
        if self.params.graph == 'dgl':
            self.selectlayerDGL()
        else:
            raise ValueError(f'Unknown graph type: {self.params.graph}')
    
    def createsecondegatlayer(self):
        if self.params.MessagePassing:
            self.egat2 = self.layer(in_node_feats=self.params.hidden_dim*self.params.num_heads,in_edge_feats=self.params.hidden_dim*self.params.num_heads,out_node_feats=self.params.hidden_dim,out_edge_feats=self.params.hidden_dim,num_heads=self.params.num_heads,edgeresid=self.params.Resid,bias=self.params.ResidBias)
        else:
            self.egat2 = nn.ModuleList()
            # Intermediate GAT layers
            for _ in range(1, self.params.egatlayers):
                self.egat2.append(self.layer(in_node_feats=self.params.hidden_dim*self.params.num_heads,in_edge_feats=self.params.hidden_dim*self.params.num_heads,out_node_feats=self.params.hidden_dim,out_edge_feats=self.params.hidden_dim,num_heads=self.params.num_heads,edgeresid=self.params.Resid,bias=self.params.ResidBias))
    
    def InitializeEGAT(self):
        self.egat1 = self.layer(in_node_feats=self.inputnodes,in_edge_feats=self.inputedges,out_node_feats=self.params.hidden_dim,out_edge_feats=self.params.hidden_dim,num_heads=self.params.num_heads,edgeresid=self.params.Resid,bias=self.params.ResidBias)
        self.createsecondegatlayer()

    def egatpass(self,graph,layer,node_feats=None,edge_feats=None):
        if node_feats is None: 
            node_feats = graph.ndata['x']
        if edge_feats is None:
            edge_feats = graph.edata['x']
        Rnode_feats, Redge_feats = layer(graph, node_feats, edge_feats)
        Rnode_feats = Rnode_feats.view(graph.number_of_nodes(),self.params.hidden_dim * self.params.num_heads)
        Redge_feats = Redge_feats.view(graph.number_of_edges(),self.params.hidden_dim * self.params.num_heads)
        return Rnode_feats, Redge_feats

    def baselayerfunction(self,graphR, layer,graphP=None,Rnode_feats=None,Redge_feats=None,Pnode_feats=None,Pedge_feats=None):
        Rnode_feats, Redge_feats = self.egatpass(graphR,layer,Rnode_feats,Redge_feats)
        R_attn_scores = self.GetAttentionMaps(graphR)
        if self.params.graph == 'reaction':
            Pnode_feats, Pedge_feats = self.egatpass(graphP,layer,Pnode_feats,Pedge_feats)
            P_attn_scores = self.GetAttentionMaps(graphP)
        else:
            Pnode_feats, Pedge_feats = None, None
            P_attn_scores = None

        return Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_scores, P_attn_scores

    def LayerOne(self, graphR, graphP=None):
        return self.baselayerfunction(graphR, self.egat1, graphP)
    
    def LayerTwoNoMP(self, graphR, graphP=None,Rnode_feats=None,Redge_feats=None,Pnode_feats=None,Pedge_feats=None):
        return self.baselayerfunction(graphR, self.egat2, graphP,Rnode_feats,Redge_feats,Pnode_feats,Pedge_feats)

    def LayerTwoMP(self, graphR, graphP=None,Rnode_feats=None,Redge_feats=None,Pnode_feats=None,Pedge_feats=None):
        R_attn_list = []
        P_attn_list = []
        for i in range(self.params.egatlayers-1):
            Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats,R_attn,P_attn = self.baselayerfunction(graphR, self.egat2, graphP,Rnode_feats,Redge_feats,Pnode_feats,Pedge_feats)
            R_attn_list.append(R_attn)
            P_attn_list.append(P_attn)
        return Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_list, P_attn_list
    
    def multiheadnormalization(self,attn_scores):
        if self.params.getattentionmaps == 'norm':
            attn_scores = torch.norm(attn_scores,dim=1)
        elif self.params.getattentionmaps == 'mean':
            attn_scores = torch.mean(attn_scores, dim=1)
        elif self.params.getattentionmaps == 'median':
            attn_scores = torch.median(attn_scores, dim=1).values
        return attn_scores
    
    def addnodescores(self,graph,attn_scores):
        if self.params.SA: 
            node_scores = graph.ndata['a_node']
            # Add node_scores to the diagonal of attn_scores for a 3D matrix
            if self.params.num_heads > 1:
                for head in range(self.params.num_heads):
                    attn_scores[:, :, head] += torch.diag(node_scores[:, head])
            else:
                attn_scores[:, :, 0] += torch.diag(node_scores[:, 0])
        return attn_scores

    def GetAttentionMaps(self,graph): #Fix this for attention scores. 
        if self.params.getattentionmaps is not None:
            attn_scores = graph.edata['a']
            attn_scores = self.addnodescores(graph,attn_scores)
            if self.params.num_heads > 1:
                attn_scores = self.multiheadnormalization(attn_scores)
            return attn_scores
        else:
            return None

    def LayerTwo(self,graphR, graphP=None,Rnode_feats=None,Redge_feats=None,Pnode_feats=None,Pedge_feats=None):
        if self.params.MessagePassing:
            return self.LayerTwoMP(graphR, graphP,Rnode_feats,Redge_feats,Pnode_feats,Pedge_feats)
        else:
            return self.LayerTwoNoMP(graphR, graphP,Rnode_feats,Redge_feats,Pnode_feats,Pedge_feats)

    def setupattndict(self):
        attn_scores = dict()
        if self.params.getattentionmaps is not None:
            attn_scores['R'] = dict()
            attn_scores['P'] = dict()
        return attn_scores
    
    def addattnmapstodict(self,attn_scores,R_attn_scores,P_attn_scores,layer='LayerOne'):
        if self.params.getattentionmaps is not None:
            attn_scores['R'][layer] = R_attn_scores
            if self.params.graph == 'reaction':
                attn_scores['P'][layer] = P_attn_scores
        return attn_scores

    def RunEGATBlock(self,graphR,graphP):
        attn_score = self.setupattndict()
        Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_scores, P_attn_scores = self.LayerOne(graphR, graphP)
        attn_score = self.addattnmapstodict(attn_score,R_attn_scores,P_attn_scores,layer='LayerOne')
        if self.params.egatlayers > 1:
            Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_scores, P_attn_scores = self.LayerTwo(graphR, graphP,Rnode_feats,Redge_feats,Pnode_feats,Pedge_feats)
            attn_score = self.addattnmapstodict(attn_score,R_attn_scores,P_attn_scores,layer='LayerTwo')
        return Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats, R_attn_scores, P_attn_scores
