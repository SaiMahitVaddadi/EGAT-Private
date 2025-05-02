import torch.nn as nn
import dgl
from .blocks.aggblock import AggregationBlock
from .blocks.egatblock import EGATBlock
from .blocks.predictionblock import PredictionBlock


class EGATModel(EGATBlock, AggregationBlock, PredictionBlock):
    def __init__(self, cfg, num_node_feats=17, num_edge_feats=14, addononlength=None, activation=None, endconvolution=None, softmax=None):
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
