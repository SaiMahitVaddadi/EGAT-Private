import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json
from egat import EGATConv,EGATConvResid,EGATConvSA,EGATConvResidSA






class ReactionEGATParams:
    # parse input parameters
    self.hidden_dim, self.num_heads = cfg.hidden_dim, cfg.num_heads
    self.aggregate = cfg.Aggregate
    self.getembeddings = cfg.Embed
    self.egatlayers = cfg.EGAT_layers

    self.selfattention = cfg.SA
    self.residualedition = cfg.Resid
    self.residbias = cfg.ResidBias

    if not cfg.useFullHyb: num_node_feats = 17
    else: num_node_feats = 22
    if not cfg.useOld: num_edge_feats = 15
    else: num_edge_feats = 14

    if cfg.removeelementinfo: num_node_feats = num_node_feats - 1

    if cfg.removeneighborcount and cfg.onlyH: num_node_feats = num_node_feats - 4
    elif cfg.removeneighborcount and not cfg.onlyH: num_node_feats = num_node_feats - 4
    elif not cfg.removeneighborcount and cfg.onlyH: num_node_feats = num_node_feats - 3
    
    if cfg.removereactiveinfo or cfg.molecular: num_node_feats = num_node_feats - 1

    if cfg.removeringinfo: 
        num_node_feats = num_node_feats - 1
        num_edge_feats = num_edge_feats - 1
    
    if cfg.removearomaticity: 
        num_node_feats = num_node_feats - 1
            
    if cfg.removeformalchargeinfo: num_node_feats = num_node_feats - 1
    if cfg.removechiralinfo: num_node_feats = num_node_feats - 3
    if cfg.removehybridinfo: num_node_feats = num_node_feats - 4
    if cfg.getradical: num_node_feats += 2
    if cfg.getspiro: num_node_feats += 1
    if cfg.getbridgehead: num_node_feats += 1
    if cfg.gethbinfo: num_node_feats += 2
    if cfg.geteneg: num_node_feats += 1
    
    if cfg.removebondorderinfo: num_edge_feats = num_edge_feats - 5
    if cfg.removebondtypeinfo or cfg.molecular: 
        if cfg.useOld:
            num_edge_feats = num_edge_feats - 4
        else:
            num_edge_feats = num_edge_feats - 5
    if cfg.removeconjinfo: num_edge_feats = num_edge_feats - 1
    if cfg.removestereoinfo: num_edge_feats = num_edge_feats - 3
    if cfg.getbondrot: num_edge_feats += 2
    self.getattentionmaps = cfg.AttentionMaps
    self.SA = cfg.SA





class ReactionEGAT(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        

        # BLOCK1: message-passing blocks

        if cfg.MessagePassing:
            if cfg.Resid is not None: 
                if not cfg.SA:
                    self.egat1 = EGATConvResid(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
                    self.egat2 = EGATConvResid(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
                else:
                    self.egat1 = EGATConvResidSA(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
                    self.egat2 = EGATConvResidSA(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
            else: 
                if not cfg.SA:   
                    self.egat1 = EGATConv(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
                    self.egat2 = EGATConv(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
                else:
                    self.egat1 = EGATConvSA(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
                    self.egat2 = EGATConvSA(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
        else:
            if cfg.Resid is not None: 
                if not cfg.SA:
                    self.egat1 = EGATConvResid(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
                    self.egat2 = nn.ModuleList()
                    # Intermediate GAT layers
                    for _ in range(1, cfg.EGAT_Layers):
                        self.egat2.append(EGATConvResid(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias))
                else:
                    self.egat1 = EGATConvResidSA(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias)
                    self.egat2 = nn.ModuleList()
                    # Intermediate GAT layers
                    for _ in range(1, cfg.EGAT_Layers):
                        self.egat2.append(EGATConvResidSA(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias))
            else: 
                if not cfg.SA:   
                    self.egat1 = EGATConv(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
                    self.egat2 = nn.ModuleList()
                    # Intermediate GAT layers
                    for _ in range(1, cfg.EGAT_Layers):
                        self.egat2.append(EGATConv(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias))
                else:
                    self.egat1 = EGATConvSA(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
                    self.egat2 = nn.ModuleList()
                    # Intermediate GAT layers
                    for _ in range(1, cfg.EGAT_Layers):
                        self.egat2.append(EGATConvSA(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads,edgeresid=cfg.Resid,bias=cfg.ResidBias))

        if cfg.Aggregate == 'Concat':
            self.agg_N_feats = nn.Sequential(nn.Linear(2*self.hidden_dim*self.num_heads, 2*self.hidden_dim*self.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Linear(2*self.hidden_dim*self.num_heads, 2*self.hidden_dim*self.num_heads, bias=True), nn.GELU())
        elif cfg.Aggregate == 'Mixing':
            self.agg_N_feats = nn.Sequential(nn.Bilinear(self.hidden_dim*self.num_heads, self.hidden_dim*self.num_heads,self.hidden_dim*self.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Bilinear(self.hidden_dim*self.num_heads, self.hidden_dim*self.num_heads,self.hidden_dim*self.num_heads, bias=True), nn.GELU())
        else:
            # BLOCK2: aggregate reactant and product nodes features
            self.agg_N_feats = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads, self.hidden_dim*self.num_heads, bias=True), nn.GELU())
            self.agg_E_feats = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads, self.hidden_dim*self.num_heads, bias=True), nn.GELU())
        

        if cfg.MixingLayer:
            if cfg.Aggregate == 'Concat':
                self.Mixing_Layer = nn.Sequential(nn.Bilinear(self.hidden_dim*self.num_heads*2,self.hidden_dim*self.num_heads*2,self.hidden_dim*self.num_heads*4,bias=True),nn.GELU())
            else:
                self.Mixing_Layer = nn.Sequential(nn.Bilinear(self.hidden_dim*self.num_heads,self.hidden_dim*self.num_heads,self.hidden_dim*self.num_heads*2,bias=True),nn.GELU())
        

        #BLOCK3: final MLP layers
        if cfg.model_type != 'BEP':
            if cfg.Aggregate == 'Concat':
                if not cfg.MixingLayer:
                    self.subnets = nn.ModuleList([PropertyNet(self.hidden_dim*self.num_heads*4, 256, 1) for _ in range(len(cfg.target))])
                else:
                    self.subnets = nn.ModuleList([PropertyNet(self.hidden_dim*self.num_heads*2, 256, 1) for _ in range(len(cfg.target))])
            else:
                self.subnets = nn.ModuleList([PropertyNet(self.hidden_dim*self.num_heads*2, 256, 1) for _ in range(len(cfg.target))])
        else:
            if cfg.Aggregate == 'Concat':
                if not cfg.MixingLayer:
                    self.subnets = nn.ModuleList([PropertyNet(self.hidden_dim*self.num_heads*4, 256, 1) for _ in range(2)])
                else:
                    self.subnets = nn.ModuleList([PropertyNet(self.hidden_dim*self.num_heads*2, 256, 1) for _ in range(2)])
            else:
                self.subnets = nn.ModuleList([PropertyNet(self.hidden_dim*self.num_heads*2, 256, 1) for _ in range(2)])

        self.MixingLayer = cfg.MixingLayer
        self.MessagePassing = cfg.MessagePassing
    def forward(self, graphR, graphP):
        ##################################### 
        ############# layer one ############# 
        ##################################### 
        Rnode_feats, Redge_feats = self.egat1(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

        Pnode_feats, Pedge_feats = self.egat1(graphP, graphP.ndata['x'], graphP.edata['x'])
        Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
        Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)

        ##################################### 
        ############# layer two ############# 
        ##################################### 
        if self.MessagePassing:
            for i in range(self.egatlayers-1):
                Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
                Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
                Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)
                
                if self.getattentionmaps and i == self.egatlayers:
                    R_attn_scores = self.egat2.edge_attn
                    if self.num_heads > 1:R_attn_scores = torch.norm(R_attn_scores,dim=1)
                    graphR.edata['norm_attn'] = R_attn_scores
                    if self.SA:
                        R_self_attn = self.egat2.self_attn
                        if self.num_heads > 1:R_self_attn = torch.norm(R_self_attn,dim=1)
                        graphR.ndata['norm_attn'] = R_self_attn
                
                    # Initialize a square matrix with zeros
                    matrix_size = graphR.number_of_nodes()
                    R_combined_matrix = torch.zeros(matrix_size, matrix_size)
                    # Fill the off-diagonal with edge attention scores
                    src, dst = graphR.edges()
                    R_combined_matrix[src, dst] = graphR.edata['norm_attn'].view(-1)

                    if self.SA:
                        # Fill the diagonal with node self-attention scores
                        R_combined_matrix.fill_diagonal_(graphR.ndata['norm_attn'].view(-1))

            
                Pnode_feats, Pedge_feats = self.egat2(graphP, Pnode_feats, Pedge_feats)
                Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
                Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)

                if self.getattentionmaps and i == self.egatlayers:
                    P_attn_scores = self.egat2.edge_attn
                    if self.num_heads > 1:P_attn_scores = torch.norm(P_attn_scores,dim=1)
                    graphP.edata['norm_attn'] = P_attn_scores
                    if self.SA:
                        P_self_attn = self.egat2.self_attn
                        if self.num_heads > 1: P_self_attn = torch.norm(P_self_attn,dim=1)
                        graphR.ndata['norm_attn'] = P_self_attn
                    
                    # Initialize a square matrix with zeros
                    matrix_size = graphP.number_of_nodes()
                    P_combined_matrix = torch.zeros(matrix_size, matrix_size)
                    # Fill the off-diagonal with edge attention scores
                    src, dst = graphP.edges()
                    P_combined_matrix[src, dst] = graphP.edata['norm_attn'].view(-1)

                    if self.SA:
                        # Fill the diagonal with node self-attention scores
                        P_combined_matrix.fill_diagonal_(graphP.ndata['norm_attn'].view(-1))
        else:
            Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
            Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
            Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

            Pnode_feats, Pedge_feats = self.egat2(graphP, Pnode_feats, Pedge_feats)
            Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
            Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)

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

        # Initialize a list to store the global features for each graph
        G_node_feats,G_edge_feats = [],[]

        # Iterate through the individual graphs
        for graph in individual_graphs:
            # Calculate the sum of the node features (assuming node features are stored in 'h')
            global_node_feature = graph.ndata['x'].sum(dim=0)
            global_edge_feature = graph.edata['x'].sum(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)

        if self.MixingLayer:
            G_features = self.Mixing_Layer(G_node_feats,G_edge_feats)
            
        else:
            G_features = torch.cat((G_node_feats,G_edge_feats), axis=1)
            
        #################################### 
        ########## merge features ##########
        #################################### 
        # MLP
        x = [subnet(G_features) for subnet in self.subnets]
        x = torch.cat(x, dim=1)

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
