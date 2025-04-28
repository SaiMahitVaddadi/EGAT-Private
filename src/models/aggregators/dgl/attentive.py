
'''
AttentiveAggregator
--------------------
A PyTorch module that implements an attention-based aggregation mechanism for graph data. 
It computes global node and edge features by learning attention scores for nodes and edges.
Attributes:
    node_attn (nn.Linear): A linear layer to compute attention scores for nodes.
    edge_attn (nn.Linear): A linear layer to compute attention scores for edges.
Methods:
    reset_parameters():
        Reinitializes the learnable parameters of the attention layers.
    forward(graph):
        Computes global node and edge features using attention mechanisms.
Example:
    >>> import dgl
    >>> import torch
    >>> from attentive import AttentiveAggregator
    >>> 
    >>> # Create a sample graph
    >>> graph = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> graph.ndata['x'] = torch.randn(4, 8)  # Node features
    >>> graph.edata['x'] = torch.randn(3, 4)  # Edge features
    >>> 
    >>> # Initialize the aggregator
    >>> aggregator = AttentiveAggregator(node_feat_size=8, edge_feat_size=4)
    >>> aggregator.reset_parameters()
    >>> 
    >>> # Forward pass
    >>> global_node_feature, global_edge_feature = aggregator(graph)
    >>> print(global_node_feature.shape, global_edge_feature.shape)
CalcAttentiveAggregator
------------------------
A PyTorch module that computes global node and edge features using precomputed attention scores 
stored in the graph's edge data.
Methods:
    forward(graph):
        Aggregates global node and edge features using precomputed attention scores.
Example:
    >>> import dgl
    >>> import torch
    >>> from attentive import CalcAttentiveAggregator
    >>> 
    >>> # Create a sample graph
    >>> graph = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> graph.ndata['x'] = torch.randn(4, 8)  # Node features
    >>> graph.edata['x'] = torch.randn(3, 1)  # Edge attention scores
    >>> 
    >>> # Initialize the aggregator
    >>> aggregator = CalcAttentiveAggregator(node_feat_size=8, edge_feat_size=1)
    >>> 
    >>> # Forward pass
    >>> global_node_feature, global_edge_feature = aggregator(graph)
    >>> print(global_node_feature.shape, global_edge_feature.shape)
CalcAttentiveAggregatorMLP
--------------------------
A PyTorch module that computes global node and edge features using an MLP-based mechanism 
to compute attention scores for edges.
Methods:
    forward(graph):
        Aggregates global node and edge features using MLP-based attention scores.
Example:
    >>> import dgl
    >>> import torch
    >>> from attentive import CalcAttentiveAggregatorMLP
    >>> 
    >>> # Create a sample graph
    >>> graph = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> graph.ndata['x'] = torch.randn(4, 8)  # Node features
    >>> graph.edata['x'] = torch.randn(3, 4)  # Edge features
    >>> 
    >>> # Initialize the aggregator
    >>> aggregator = CalcAttentiveAggregatorMLP(node_feat_size=8, edge_feat_size=4)
    >>> 
    >>> # Forward pass
    >>> global_node_feature, global_edge_feature = aggregator(graph)
    >>> print(global_node_feature.shape, global_edge_feature.shape)
'''
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
    

    def forward(self, graph):
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
        
    def forward(self, graph):
        # Edge attention scores
        if graph.edata['x'].dim() == 2:  # Single attention head
            edge_scores = graph.edata['a']  # [E, 1]
        else:  # Multi-head attention
            edge_scores = graph.edata['a'].mean(dim=1, keepdim=True)  # [E, 1]

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
        self.edge_learning = nn.Sequential(
                nn.Linear(graph.edata['a'].size(-1), 16),  # Hidden layer with 16 units
                nn.ReLU(),
                nn.Linear(16, 1)  # Output layer with 1 unit
            )
        
    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.edge_learning[0].weight)
        nn.init.zeros_(self.edge_learning[0].bias)
        nn.init.xavier_uniform_(self.edge_learning[2].weight)
        nn.init.zeros_(self.edge_learning[2].bias)
        
    def forward(self, graph):
        # Edge attention scores
        if graph.edata['a'].dim() == 1:  # Single attention head
            edge_scores = graph.edata['a']  # [E, 1]
        else:  # Multi-head attention
            edge_scores = self.edge_learning(graph.edata['a'])  # [E, 1]

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



if __name__ == '__main__':
    # Create a sample graph
    graph = dgl.graph(([0, 1, 2], [1, 2, 3]))
    graph.ndata['x'] = torch.randn(4, 8)  # Node features
    graph.edata['x'] = torch.randn(3, 4)  # Edge features
    graph.edata['a'] = torch.randn(3, 1)  # Edge attention scores
    
    # Initialize the aggregator
    aggregator = AttentiveAggregator(node_feat_size=8, edge_feat_size=4)
    aggregator.reset_parameters()
    
    # Forward pass
    global_node_feature, global_edge_feature = aggregator(graph)
    print(global_node_feature, global_edge_feature)

    # Initialize the calculator
    calc_aggregator = CalcAttentiveAggregator(node_feat_size=8, edge_feat_size=1)
    # Forward pass
    global_node_feature, global_edge_feature = calc_aggregator(graph)
    print(global_node_feature, global_edge_feature)

    # Initialize the calculator
    calc_aggregator_mlp = CalcAttentiveAggregatorMLP(node_feat_size=8, edge_feat_size=4)
    calc_aggregator_mlp.reset_parameters()
    # Forward pass
    global_node_feature, global_edge_feature = calc_aggregator_mlp(graph)
    print(global_node_feature.shape, global_edge_feature.shape)
    