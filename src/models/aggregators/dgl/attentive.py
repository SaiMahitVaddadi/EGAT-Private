import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class AttentiveAggregator(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        # Attention networks for nodes and edges
        self.node_attn = nn.Linear(node_feat_size, 1)  # Learns node importance
        self.edge_attn = nn.Linear(edge_feat_size, 1)  # Learns edge importance

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.node_attn.weight)
        nn.init.zeros_(self.node_attn.bias)
        nn.init.xavier_uniform_(self.edge_attn.weight)
        nn.init.zeros_(self.edge_attn.bias)
    

    def forward(self, g):
        # --- Node Attention ---
        node_feats = graph.ndata['x']
        # Compute attention scores (logits)
        node_logits = self.node_attn(node_feats).squeeze()  # [N] Write out the node_attn function
        node_e = node_logits.exp()                          # [N]
        node_z = node_e.sum()                               # Scalar (sum for normalization)
        node_alphas = node_e / node_z                       # [N]
        # Weighted sum of node features
        global_node_feature = (node_alphas.unsqueeze(-1) * node_feats).sum(dim=0)  # [d_node]

        # --- Edge Attention ---
        edge_feats = graph.edata['x']
        # Compute attention scores (logits)
        edge_logits = self.edge_attn(edge_feats).squeeze()  # [E] Write out the edge_attn function 
        edge_e = edge_logits.exp()                          # [E]
        edge_z = edge_e.sum()                               # Scalar
        edge_alphas = edge_e / edge_z                       # [E]
        # Weighted sum of edge features
        global_edge_feature = (edge_alphas.unsqueeze(-1) * edge_feats).sum(dim=0)  # [d_edge]

        return global_node_feature, global_edge_feature
    

class CalcAttentiveAggregator(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        
    def forward(self, g):
        # Edge attention scores
        if graph.edata['x'].dim() == 2:  # Single attention head
            edge_scores = graph.edata['x']  # [E, 1]
        else:  # Multi-head attention
            edge_scores = graph.edata['x'].mean(dim=1, keepdim=True)  # [E, 1]

        # Node attention scores
        with graph.local_scope():
            graph.edata['edge_scores'] = edge_scores
            graph.update_all(fn.copy_e('edge_scores', 'm'), fn.mean('m', 'node_scores'))
            node_scores = graph.ndata['node_scores']  # [N, 1]

        # Weighted node aggregation
        weighted_nodes = graph.ndata['x'] * node_scores  # [N, d_node] ⊙ [N, 1]
        global_node_feature = weighted_nodes.sum(dim=0)  # [d_node]

        # Weighted edge aggregation
        weighted_edges = graph.edata['x'] * edge_scores  # [E, d_edge] ⊙ [E, 1]
        global_edge_feature = weighted_edges.sum(dim=0)  # [d_edge]
            
        return global_node_feature, global_edge_feature
class CalcAttentiveAggregatorMLP(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        
    def forward(self, g):
        
        # Edge attention scores
        if graph.edata['x'].dim() == 2:  # Single attention head
            edge_scores = graph.edata['x']  # [E, 1]
        else:  # Multi-head attention
            edge_scores = nn.Sequential(
                nn.Linear(graph.edata['x'].size(-1), 16),  # Hidden layer with 16 units
                nn.ReLU(),
                nn.Linear(16, 1)  # Output layer with 1 unit
            )(graph.edata['x'])  # [E, 1]

        # Node attention scores
        with graph.local_scope():
            graph.edata['edge_scores'] = edge_scores
            graph.update_all(fn.copy_e('edge_scores', 'm'), fn.sum('m', 'sum_edge_scores'))
            graph.update_all(fn.copy_e('edge_scores', 'm'), fn.mean('m', 'mean_edge_scores'))
            sum_edge_scores = graph.ndata['sum_edge_scores']  # [N, 1]
            mean_edge_scores = graph.ndata['mean_edge_scores']  # [N, 1]
            node_scores = sum_edge_scores / mean_edge_scores  # [N, 1]

        # Weighted node aggregation
        weighted_nodes = graph.ndata['x'] * node_scores  # [N, d_node] ⊙ [N, 1]
        global_node_feature = weighted_nodes.sum(dim=0)  # [d_node]

        # Weighted edge aggregation
        weighted_edges = graph.edata['x'] * edge_scores  # [E, d_edge] ⊙ [E, 1]
        global_edge_feature = weighted_edges.sum(dim=0)  # [d_edge]
            
        return global_node_feature, global_edge_feature