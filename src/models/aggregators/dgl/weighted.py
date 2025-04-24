import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class WeightedSumAggregator(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        # Learnable weights for nodes and edges
        self.node_weights = nn.Parameter(torch.randn(node_feat_size))  # [d_node]
        self.edge_weights = nn.Parameter(torch.randn(edge_feat_size))  # [d_edge]

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.node_weights)
        nn.init.xavier_uniform_(self.edge_weights)

    def forward(self, graph):
        
        # Weighted node aggregation
        weighted_nodes = graph.ndata['x'] * self.node_weights  # [N, d_node] ⊙ [d_node]
        global_node_feature = weighted_nodes.sum(dim=0)        # [d_node]
        
        # Weighted edge aggregation
        weighted_edges = graph.edata['x'] * self.edge_weights  # [E, d_edge] ⊙ [d_edge]
        global_edge_feature = weighted_edges.sum(dim=0)        # [d_edge]
        return global_node_feature, global_edge_feature

class WeightedMeanAggregator(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        # Learnable weights for nodes and edges
        self.node_weights = nn.Parameter(torch.randn(node_feat_size))  # [d_node]
        self.edge_weights = nn.Parameter(torch.randn(edge_feat_size))  # [d_edge]

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.node_weights)
        nn.init.xavier_uniform_(self.edge_weights)

    def forward(self, graph):
        # Weighted node aggregation
        weighted_nodes = graph.ndata['x'] * self.node_weights  # [N, d_node] ⊙ [d_node]
        global_node_feature = weighted_nodes.mean(dim=0)       # [d_node]
        
        # Weighted edge aggregation
        weighted_edges = graph.edata['x'] * self.edge_weights  # [E, d_edge] ⊙ [d_edge]
        global_edge_feature = weighted_edges.mean(dim=0)       # [d_edge]
        
        return global_node_feature, global_edge_feature
    
class BasicPooledAggregator(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        
        self.aggregators ['sum','mean','max','min','std']
    
    
    def _dglfcn(self,graph,aggregator='sum'):
        global_node_feature = getattr(graph.ndata['x'],aggregator)(dim=0)
        global_edge_feature = getattr(graph.edata['x'],aggregator)(dim=0)
        return global_node_feature, global_edge_feature

    def forward(self,graph):
        node_features,edge_features = [],[] 
        for aggregator in self.aggregators:
            node_feature,edge_feature = self._dglfcn(graph,aggregator)
            node_features.append(node_feature)
            edge_features.append(edge_feature)
        
        global_node_feature = torch.cat(node_features, dim=0)
        global_edge_feature = torch.cat(edge_features, dim=0)
        return global_node_feature, global_edge_feature
    
class LearnedPooledAggregator(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        
        self.aggregators ['sum','mean','max','min','std']
    
    
    def _dglfcn(self,graph,aggregator='sum'):
        global_node_feature = getattr(graph.ndata['x'],aggregator)(dim=0)
        global_edge_feature = getattr(graph.edata['x'],aggregator)(dim=0)
        return global_node_feature, global_edge_feature

    def forward(self,graph):
        node_features,edge_features = [],[] 
        for aggregator in self.aggregators:
            node_feature,edge_feature = self._dglfcn(graph,aggregator)
            node_features.append(node_feature)
            edge_features.append(edge_feature)
        
        global_node_feature = torch.cat(node_features, dim=0)
        global_edge_feature = torch.cat(edge_features, dim=0)

        # Linear layers to condense features to 512-length vectors
        self.node_linear = nn.Linear(global_node_feature.size(0), 512)
        self.edge_linear = nn.Linear(global_edge_feature.size(0), 512)

        global_node_feature = self.node_linear(global_node_feature)
        global_edge_feature = self.edge_linear(global_edge_feature)

        return global_node_feature, global_edge_feature
    

class LearnedPooledAggregatorwithWeightedFeatures(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        # Learnable weights for nodes and edges
        self.node_weights_mean = nn.Parameter(torch.randn(node_feat_size))  # [d_node]
        self.edge_weights_mean = nn.Parameter(torch.randn(edge_feat_size))  # [d_edge]
        self.node_weights_max = nn.Parameter(torch.randn(node_feat_size))
        self.edge_weights_max = nn.Parameter(torch.randn(edge_feat_size))
        self.node_weights_min = nn.Parameter(torch.randn(node_feat_size))
        self.edge_weights_min = nn.Parameter(torch.randn(edge_feat_size))
        self.node_weights_std = nn.Parameter(torch.randn(node_feat_size))
        self.edge_weights_std = nn.Parameter(torch.randn(edge_feat_size))
        self.node_weights_sum = nn.Parameter(torch.randn(node_feat_size))  # [d_node]
        self.edge_weights_sum = nn.Parameter(torch.randn(edge_feat_size))  # [d_edge]
        self.aggregators ['sum','mean','max','min','std']
    
    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.node_weights_mean)
        nn.init.xavier_uniform_(self.edge_weights_mean)
        nn.init.xavier_uniform_(self.node_weights_max)
        nn.init.xavier_uniform_(self.edge_weights_max)
        nn.init.xavier_uniform_(self.node_weights_min)
        nn.init.xavier_uniform_(self.edge_weights_min)
        nn.init.xavier_uniform_(self.node_weights_std)
        nn.init.xavier_uniform_(self.edge_weights_std)
        nn.init.xavier_uniform_(self.node_weights_sum)
        nn.init.xavier_uniform_(self.edge_weights_sum)
    
    def _dglfcn(self,graph,aggregator='sum'):
        global_node_feature = getattr(graph.ndata['x'],aggregator)(dim=0)
        global_edge_feature = getattr(graph.edata['x'],aggregator)(dim=0)
        return global_node_feature, global_edge_feature
    
    def _dglweightedfcn(self,graph,aggregator='sum'):
        weighted_nodes = graph.ndata['x'] * getattr(self,f'node_weights_{aggregator}')  # [N, d_node] ⊙ [d_node]
        global_node_feature = getattr(weighted_nodes,aggregator)(dim=0)
        weighted_edges = graph.edata['x'] * getattr(self,f'edge_weights_{aggregator}')
        global_edge_feature = getattr(weighted_edges,aggregator)(dim=0)
        return global_node_feature, global_edge_feature


    def forward(self,graph):
        node_features,edge_features = [],[] 
        for aggregator in self.aggregators:
            node_feature,edge_feature = self._dglfcn(graph,aggregator)
            node_features.append(node_feature)
            edge_features.append(edge_feature)
            node_feature,edge_feature = self._dglweightedfcn(graph,aggregator)
            node_features.append(node_feature)
            edge_features.append(edge_feature)
        
        global_node_feature = torch.cat(node_features, dim=0)
        global_edge_feature = torch.cat(edge_features, dim=0)
        # Linear layers to condense features to 512-length vectors
        self.node_linear = nn.Linear(global_node_feature.size(0), 512)
        self.edge_linear = nn.Linear(global_edge_feature.size(0), 512)

        global_node_feature = self.node_linear(global_node_feature)
        global_edge_feature = self.edge_linear(global_edge_feature)

        return global_node_feature, global_edge_feature

        