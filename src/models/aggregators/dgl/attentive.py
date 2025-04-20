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
    

    def forward(self, individual_graphs):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
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
            
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)
        
        return torch.stack(G_node_feats), torch.stack(G_edge_feats)
    

class CalcAttentiveAggregator(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int):
        super().__init__()
        
    def forward(self, individual_graphs):
        G_node_feats, G_edge_feats = [], []
        for graph in individual_graphs:
            # Node attention scores
            node_scores = torch.softmax(graph.ndata['x'], dim=0).sum(dim=1, keepdim=True)  # [N, 1]
            # Edge attention scores
            edge_scores = torch.softmax(graph.edata['x'], dim=0).sum(dim=1, keepdim=True)  # [E, 1]

            # Weighted node aggregation
            weighted_nodes = graph.ndata['x'] * node_scores  # [N, d_node] ⊙ [N, 1]
            global_node_feature = weighted_nodes.sum(dim=0)  # [d_node]

            # Weighted edge aggregation
            weighted_edges = graph.edata['x'] * edge_scores  # [E, d_edge] ⊙ [E, 1]
            global_edge_feature = weighted_edges.sum(dim=0)  # [d_edge]
            
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)
        
        return torch.stack(G_node_feats), torch.stack(G_edge_feats)