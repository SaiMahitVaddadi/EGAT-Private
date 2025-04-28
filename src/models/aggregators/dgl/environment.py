import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv


class AggregatorFunctions(nn.Module):
    def __init__(self,node_agg=None,edge_agg=None,mixing=None,node_feat_size: int = 0, edge_feat_size: int = 0,global_agg='sum'):
        super().__init__()
        self.setup(node_agg, edge_agg, mixing, node_feat_size, edge_feat_size,global_agg)


    def setup(self, node_agg=None, edge_agg=None, mixing=None, node_feat_size: int = 0, edge_feat_size: int = 0,global_agg='sum'):
        self.node_agg = node_agg
        self.edge_agg = edge_agg
        self.global_agg = global_agg
        self.mixing = mixing
        if mixing == 'linear':
            self.mixinglayer = nn.Linear(node_feat_size + edge_feat_size, node_feat_size + edge_feat_size)
        elif mixing == 'bilinear':
            self.mixinglayer = nn.Bilinear(node_feat_size, edge_feat_size, node_feat_size + edge_feat_size)
    
    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        if self.node_agg == 'weighted_sum':
            nn.init.xavier_uniform_(self.node_weights)
        elif self.node_agg == 'learned_attn':
            nn.init.xavier_uniform_(self.node_attn.weight)
            nn.init.zeros_(self.node_attn.bias)
        
        if self.edge_agg == 'weighted_sum':
            nn.init.xavier_uniform_(self.edge_weights)
        elif self.edge_agg == 'learned_attn':
            nn.init.xavier_uniform_(self.edge_attn.weight)
            nn.init.zeros_(self.edge_attn.bias)

    def getnodeaggfcns(self,node_feat_size: int = 0):
        if self.node_agg == 'weighted_sum':
            # Learnable weights for nodes and edges
            self.node_weights = nn.Parameter(torch.randn(node_feat_size))  # [d_node]
        elif self.node_agg == 'learned_attn':
            self.node_attn = nn.Linear(node_feat_size, 1)  # Learns node importance
        

    def getedgeaddfcns(self,edge_feat_size: int = 0):
        if self.edge_agg == 'weighted_sum':
            # Learnable weights for nodes and edges
            self.edge_weights = nn.Parameter(torch.randn(edge_feat_size))
        elif self.edge_agg == 'learned_attn':
            self.edge_attn = nn.Linear(edge_feat_size, 1)
    

    def weighted_sum_node(self,graph):
        weighted_nodes = graph.ndata['x'] * self.node_weights  # [N, d_node] ⊙ [d_node]
        global_node_feature = weighted_nodes.sum(dim=0)        # [d_node]
    
        return global_node_feature
    
    def weighted_sum_edge(self,graph):
        # Weighted edge aggregation
        weighted_edges = graph.edata['x'] * self.edge_weights  # [E, d_edge] ⊙ [d_edge]
        global_edge_feature = weighted_edges.sum(dim=0)        # [d_edge]
        return global_edge_feature
    

    def learned_attn_node(self,graph):
        # --- Node Attention ---
        node_feats = graph.ndata['x']
        # Compute attention scores (logits)
        node_logits = self.node_attn(node_feats).squeeze()  # [N] Write out the node_attn function
        node_e = node_logits.exp()                          # [N]
        node_z = node_e.sum()                               # Scalar (sum for normalization)
        node_alphas = node_e / node_z                       # [N]
        # Weighted sum of node features
        global_node_feature = (node_alphas.unsqueeze(-1) * node_feats).sum(dim=0)  # [d_node]
        return global_node_feature
    
    def learned_attn_edge(self,graph):
        # --- Edge Attention ---
        edge_feats = graph.edata['x']
        # Compute attention scores (logits)
        edge_logits = self.edge_attn(edge_feats).squeeze()  # [E] Write out the edge_attn function 
        edge_e = edge_logits.exp()                          # [E]
        edge_z = edge_e.sum()                               # Scalar
        edge_alphas = edge_e / edge_z                       # [E]
        # Weighted sum of edge features
        global_edge_feature = (edge_alphas.unsqueeze(-1) * edge_feats).sum(dim=0)  # [d_edge]

        return global_edge_feature
    
    def calced_attn_node(self,graph):
        # Node attention scores
        node_scores = torch.softmax(graph.ndata['x'], dim=0).sum(dim=1, keepdim=True)  # [N, 1]
        

        # Weighted node aggregation
        weighted_nodes = graph.ndata['x'] * node_scores  # [N, d_node] ⊙ [N, 1]
        global_node_feature = weighted_nodes.sum(dim=0)  # [d_node]
        return global_node_feature
    
    def calced_attn_edge(self,graph):
        # Edge attention scores
        edge_scores = torch.softmax(graph.edata['x'], dim=0).sum(dim=1, keepdim=True)  # [E, 1]
        # Weighted edge aggregation
        weighted_edges = graph.edata['x'] * edge_scores  # [E, d_edge] ⊙ [E, 1]
        global_edge_feature = weighted_edges.sum(dim=0)  # [d_edge]
        return global_edge_feature
    
    def existing_attn_node(self,graph):
        # Get neighboring nodes h_j and their attention scores a_ij
        graph.apply_edges(fn.u_mul_e('x', 'a', 'm'))  # Multiply h_j with a_ij
        # Add h_i with the sum of a_ij * h_j
        graph.update_all(fn.u_mul_e('x', 'a', 'm'), fn.sum('m', 'h_env'))  # Compute sum of a_ij * h_j
        graph.ndata['h_env'] += graph.ndata['x']  # Add h_i to the aggregated features
        return graph.ndata['h_env']
    
    def existing_attn_edge(self,graph):
        # Get all edges e_ij for node h_i and multiply with attention score a_ij
        graph.apply_edges(fn.e_mul_e('x', 'a', 'm'))  # Multiply e_ij with a_ij
        # Sum the weighted edges for each node
        graph.update_all(fn.e_mul_e('x', 'a', 'm'), fn.sum('m', 'f_env'))  # Compute sum of a_ij * e_ij
        return graph.edata['f_env']


    def resetlayers(self):
        """
        Reinitialize learnable parameters.
        """
        if self.node_agg == 'weighted_sum':
            nn.init.xavier_uniform_(self.node_weights)
        elif self.node_agg == 'learned_attn':
            nn.init.xavier_uniform_(self.node_attn.weight)
            nn.init.zeros_(self.node_attn.bias)
        
        if self.edge_agg == 'weighted_sum':
            nn.init.xavier_uniform_(self.edge_weights)
        elif self.edge_agg == 'learned_attn':
            nn.init.xavier_uniform_(self.edge_attn.weight)
            nn.init.zeros_(self.edge_attn.bias)
        if self.mixing == 'linear':
            nn.init.xavier_uniform_(self.mixinglayer.weight)
            nn.init.zeros_(self.mixinglayer.bias)
        elif self.mixing == 'bilinear':
            nn.init.xavier_uniform_(self.mixinglayer.weight)
            nn.init.zeros_(self.mixinglayer.bias)

    def nodeaggstatement(self,g):
        # Step 1: Aggregate neighboring node features (sum)
        if self.node_agg is None: g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_env'))
        elif self.node_agg == 'mean': g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_env'))
        elif self.node_agg == 'max': g.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h_env'))
        elif self.node_agg == 'min': g.update_all(fn.copy_u('h', 'm'), fn.min('m', 'h_env'))
        elif self.node_agg == 'mean': g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_env'))
        elif self.node_agg == 'weighted_sum': g.ndata['h_env'] = self.weighted_sum_node(g)
        elif self.node_agg == 'learned_attn': g.ndata['h_env'] = self.learned_attn_node(g)
        elif self.node_agg == 'calced_attn': g.ndata['h_env'] = self.calced_attn_node(g)
        elif self.node_agg == 'norm': 
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_env'))
            g.ndata['h_env'] = g.ndata['h_env'] / torch.norm(g.ndata['h_env'], dim=1, keepdim=True)
        else: raise NotImplementedError("Node aggregation not implemented")

    def edgeaggstatement(self,g):
        # Step 2: Aggregate neighboring edge features (sum)
        if self.edge_agg == None: g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'f_env'))
        elif self.edge_agg == 'mean': g.update_all(fn.copy_e('e', 'm'), fn.mean('m', 'f_env'))
        elif self.edge_agg == 'max': g.update_all(fn.copy_e('e', 'm'), fn.max('m', 'f_env'))
        elif self.edge_agg == 'min': g.update_all(fn.copy_e('e', 'm'), fn.min('m', 'f_env'))
        elif self.edge_agg == 'sum': g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'f_env'))
        elif self.edge_agg == 'weighted_sum': g.edata['f_env'] = self.weighted_sum_edge(g)
        elif self.edge_agg == 'learned_attn': g.edata['f_env'] = self.learned_attn_edge(g)
        elif self.edge_agg == 'calced_attn': g.edata['f_env'] = self.calced_attn_edge(g)
        elif self.edge_agg == 'norm':
            g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'f_env'))
            g.edata['f_env'] = g.ndata['f_env'] / torch.norm(g.ndata['f_env'], dim=1, keepdim=True)
        else: raise NotImplementedError("Edge aggregation not implemented")

    def concatenvfeatures(self,g):
        # Step 3: Concatenate environment features
        h_env = g.ndata['h_env']
        e_env = g.ndata['f_env']
        if self.mixing is None:
            a_env = torch.cat([h_env, e_env], dim=1)
        elif self.mixing == 'linear':
            a_env = torch.cat([h_env, e_env], dim=1)
            a_env = self.mixinglayer(a_env)
        elif self.mixing == 'bilinear':
            a_env = self.mixinglayer(h_env, e_env)

        g.ndata['a_env'] = a_env
        return a_env
    
    def globalaggstatement(self,a_env):
        # Step 4: Global sum pooling [add the weighted pooling and the attention pooling for the GAT]
        if self.global_agg == 'sum':
            graph_rep = torch.sum(a_env, dim=0, keepdim=True)
        elif self.global_agg == 'mean':
            graph_rep = torch.mean(a_env, dim=0, keepdim=True)
        elif self.global_agg == 'max':
            graph_rep = torch.max(a_env, dim=0, keepdim=True).values
        elif self.global_agg == 'min':
            graph_rep = torch.min(a_env, dim=0, keepdim=True).values
        elif self.global_agg == 'norm':
            graph_rep = torch.norm(a_env, dim=0, keepdim=True)
        elif self.global_agg == 'std':
            graph_rep = torch.std(a_env, dim=0, keepdim=True)
        else: raise NotImplementedError("Global aggregation not implemented")
        return graph_rep


class EnvironmentAggregator(AggregatorFunctions):
    def __init__(self,node_agg=None,edge_agg=None,mixing=None,node_feat_size: int = 0, edge_feat_size: int = 0,global_agg='sum'):
        super().__init__()
        self.setup(node_agg, edge_agg, mixing, node_feat_size, edge_feat_size,global_agg)
        
    def reset_parameters(self):
        self.resetlayers()

            
    def aggregate(self, g):
        """
        Perform environment aggregation for all nodes in the graph
        
        Args:
            g (DGLGraph): Input graph with node features 'h' and edge features 'e'
            
        Returns:
            a_env (Tensor): Aggregated node features [num_nodes, node_dim + edge_dim]
            graph_rep (Tensor): Global sum pooled representation [1, node_dim + edge_dim]
        """
        self.nodeaggstatement(g)
        self.edgeaggstatement(g)
        a_env = self.concatenvfeatures(g)
        # Step 4: Global sum pooling [ fix to do any type]
        graph_rep = self.globalaggstatement(a_env)
        return a_env, graph_rep
    
    

    def forward(self, g):
        """
        Perform aggregation for the given graph using the specified node and edge aggregation methods.

        Args:
            g (DGLGraph): Input graph with node features 'h' and edge features 'e'.

        Returns:
            a_env (Tensor): Aggregated node and edge features [num_nodes, node_dim + edge_dim].
            graph_rep (Tensor): Global representation of the graph [1, node_dim + edge_dim].
        """
        # Perform aggregation
        a_env, graph_rep = self.aggregate(g)

    
        return a_env, graph_rep

class EnvironmentAggregatorwithGAT(AggregatorFunctions):
    def __init__(self,node_agg=None,edge_agg=None,mixing=None,node_feat_size: int = 0, edge_feat_size: int = 0,global_agg='sum'):
        super().__init__()
        self.setup(node_agg, edge_agg, mixing, node_feat_size, edge_feat_size,global_agg)
        # GAT layer for node aggregation
        self.gat = GATConv(node_feat_size+edge_feat_size, node_feat_size+edge_feat_size, num_heads=1, feat_drop=0.6, attn_drop=0.6, negative_slope=0.2, residual=False)
        
    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        self.resetlayers()
        self.gat.reset_parameters()
                
    def aggregate(self, g):
        """
        Perform environment aggregation for all nodes in the graph
        
        Args:
            g (DGLGraph): Input graph with node features 'h' and edge features 'e'
            
        Returns:
            a_env (Tensor): Aggregated node features [num_nodes, node_dim + edge_dim]
            graph_rep (Tensor): Global sum pooled representation [1, node_dim + edge_dim]
        """

        # Step 1: Aggregate neighboring node features (sum)
        self.nodeaggstatement(g)
        # Step 2: Aggregate neighboring edge features (sum)
        self.edgeaggstatement(g)
        # Step 3: Concatenate environment features
        a_env = self.concatenvfeatures(g)  
        g = dgl.add_self_loop(g)
        a_env = self.gat(g, a_env)
        #Step 5: Use your favoite pooling method
        graph_rep = self.globalaggstatement(a_env)        
        return a_env, graph_rep
    
    

    def forward(self, g):
        """
        Perform aggregation for the given graph using the specified node and edge aggregation methods.

        Args:
            g (DGLGraph): Input graph with node features 'h' and edge features 'e'.

        Returns:
            a_env (Tensor): Aggregated node and edge features [num_nodes, node_dim + edge_dim].
            graph_rep (Tensor): Global representation of the graph [1, node_dim + edge_dim].
        """
        # Perform aggregation
        a_env, graph_rep = self.aggregate(g)

    
        return a_env, graph_rep
if __name__ == '__main__':
    # Create a sample graph
    import dgl
    import torch

    # Create a sample graph
    g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    g.ndata['h'] = torch.randn(4, 8)  # Node features
    g.edata['e'] = torch.randn(3, 4)  # Edge features

    # Initialize the aggregator
    aggregator = EnvironmentAggregator(node_agg='mean', edge_agg='sum', mixing='linear', node_feat_size=8, edge_feat_size=4)
    aggregator.reset_parameters()

    # Forward pass
    a_env, graph_rep = aggregator(g)
    print(a_env, graph_rep)

    # Initialize the aggregator with GAT
    aggregator_gat = EnvironmentAggregatorwithGAT(node_agg='mean', edge_agg='mean', mixing='linear', node_feat_size=8, edge_feat_size=4)
    aggregator_gat.reset_parameters()
    # Forward pass
    a_env_gat, graph_rep_gat = aggregator_gat(g)
    print(a_env_gat, graph_rep_gat)