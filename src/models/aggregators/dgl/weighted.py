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

    def forward(self, individual_graphs):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
            # Weighted node aggregation
            weighted_nodes = graph.ndata['x'] * self.node_weights  # [N, d_node] ⊙ [d_node]
            global_node_feature = weighted_nodes.sum(dim=0)        # [d_node]
            
            # Weighted edge aggregation
            weighted_edges = graph.edata['x'] * self.edge_weights  # [E, d_edge] ⊙ [d_edge]
            global_edge_feature = weighted_edges.sum(dim=0)        # [d_edge]
            
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)
        
        return torch.stack(G_node_feats), torch.stack(G_edge_feats)


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

    def forward(self, individual_graphs):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
            # Weighted node aggregation
            weighted_nodes = graph.ndata['x'] * self.node_weights  # [N, d_node] ⊙ [d_node]
            global_node_feature = weighted_nodes.mean(dim=0)       # [d_node]
            
            # Weighted edge aggregation
            weighted_edges = graph.edata['x'] * self.edge_weights  # [E, d_edge] ⊙ [d_edge]
            global_edge_feature = weighted_edges.mean(dim=0)       # [d_edge]
            
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)
        
        return torch.stack(G_node_feats), torch.stack(G_edge_feats)