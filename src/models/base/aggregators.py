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
    



class EnvironmentAggregator:
    def __init__(self,node_agg=None,edge_agg=None,mixing=None,node_feat_size: int = 0, edge_feat_size: int = 0):
        self.node_agg = node_agg
        self.edge_agg = edge_agg
        self.mixing = mixing
        if mixing == 'linear':
            self.mixinglayer = nn.Linear(node_feat_size + edge_feat_size, node_feat_size + edge_feat_size)
        elif mixing == 'bilinear':
            self.mixinglayer = nn.Bilinear(node_feat_size, edge_feat_size, node_feat_size + edge_feat_size)
        
        
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
        if self.node_agg is None: g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h_env'))
        elif self.node_agg == 'mean': g.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'h_env'))
        elif self.node_agg == 'max': g.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h_env'))
        elif self.node_agg == 'min': g.update_all(fn.copy_src('h', 'm'), fn.min('m', 'h_env'))
        elif self.node_agg == 'mean': g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h_env'))
        elif self.node_agg == 'weighted_sum': g.ndata['h_env'] = self.weighted_sum_node(g)
        elif self.node_agg == 'learned_attn': g.ndata['h_env'] = self.learned_attn_node(g)
        elif self.node_agg == 'calced_attn': g.ndata['h_env'] = self.calced_attn_node(g)
        elif self.node_agg == 'norm': 
            g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h_env'))
            g.ndata['h_env'] = g.ndata['h_env'] / torch.norm(g.ndata['h_env'], dim=1, keepdim=True)
        else: raise NotImplementedError("Node aggregation not implemented")

        # Step 2: Aggregate neighboring edge features (sum)
        if self.edge_agg is None: g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'f_env'))
        elif self.edge_agg == 'mean': g.update_all(fn.copy_edge('e', 'm'), fn.mean('m', 'f_env'))
        elif self.edge_agg == 'max': g.update_all(fn.copy_edge('e', 'm'), fn.max('m', 'f_env'))
        elif self.edge_agg == 'min': g.update_all(fn.copy_edge('e', 'm'), fn.min('m', 'f_env'))
        elif self.edge_agg == 'mean': g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'f_env'))
        elif self.edge_agg == 'weighted_sum': g.edata['f_env'] = self.weighted_sum_edge(g)
        elif self.edge_agg == 'learned_attn': g.edata['f_env'] = self.learned_attn_edge(g)
        elif self.edge_agg == 'calced_attn': g.edata['f_env'] = self.calced_attn_edge(g)
        elif self.edge_agg == 'norm':
            g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'f_env'))
            g.edata['f_env'] = g.edata['f_env'] / torch.norm(g.edata['f_env'], dim=1, keepdim=True)
        else: raise NotImplementedError("Edge aggregation not implemented")



        
        # Step 3: Concatenate environment features
        h_env = g.ndata['h_env']
        e_env = g.ndata['f_env']
        if self.mixing is None:
            a_env = torch.cat([h_env, e_env], dim=1)
        elif self.mixing == 'linear':
            a_env = self.mixinglayer(a_env)
        elif self.mixing == 'bilinear':
            a_env = self.mixinglayer(h_env, e_env)

        
        # Step 4: Global sum pooling [ fix to do any type]
        graph_rep = torch.sum(a_env, dim=0, keepdim=True)
        
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

class BondAggregator:
    def __init__(self,use_nn=False,node_feat_size: int = 0, edge_feat_size: int = 0):
        if use_nn : self.nnlayer = nn.Linear(2*node_feat_size, edge_feat_size)
        pass

    def forward(self,g):
        # Get node and edge features
        node_features = g.ndata['h']  # Node features [num_nodes, node_dim]
        edge_features = g.edata['e']  # Edge features [num_edges, edge_dim]

        # Combine node and edge features for each edge
        src, dst = g.edges()  # Get source and destination node indices for each edge
        src_node_features = node_features[src]  # Source node features [num_edges, node_dim]
        dst_node_features = node_features[dst]  # Destination node features [num_edges, node_dim]

        # Concatenate source, destination, and edge features
        combined_features = torch.cat([src_node_features, dst_node_features, edge_features], dim=1)  # [num_edges, 2 * node_dim + edge_dim]

        # Apply neural network layer if specified
        if self.use_nn:
            combined_features = self.nnlayer(combined_features)

        

        # Pool across all edges based on the specified aggregation method
        if self.node_agg == 'mean':
            pooled_features = combined_features.mean(dim=0, keepdim=True)  # Mean pooling
        elif self.node_agg == 'sum':
            pooled_features = combined_features.sum(dim=0, keepdim=True)  # Sum pooling
        elif self.node_agg == 'max':
            pooled_features = combined_features.max(dim=0, keepdim=True).values  # Max pooling
        elif self.node_agg == 'min':
            pooled_features = combined_features.min(dim=0, keepdim=True).values  # Min pooling
        elif self.node_agg == 'norm':
            pooled_features = combined_features / torch.norm(combined_features, dim=1, keepdim=True).sum(dim=0, keepdim=True)  # Normalized pooling
        else:
            raise NotImplementedError(f"Pooling method '{self.node_agg}' not implemented")

        return pooled_features



class BondEnvironmentAggregator:
    def __init__(self,use_nn=False,node_feat_size: int = 0, edge_feat_size: int = 0):
        if use_nn : self.nnlayer = nn.Linear(2*node_feat_size, edge_feat_size)
        pass

    def forward(self,g):
        # Get node and edge features
        node_features = g.ndata['h']  # Node features [num_nodes, node_dim]
        edge_features = g.edata['e']  # Edge features [num_edges, edge_dim]

        # Combine node and edge features for each edge
        src, dst = g.edges()  # Get source and destination node indices for each edge
        src_node_features = node_features[src]  # Source node features [num_edges, node_dim]
        dst_node_features = node_features[dst]  # Destination node features [num_edges, node_dim]

        # Concatenate source, destination, and edge features
        combined_features = torch.cat([src_node_features, dst_node_features, edge_features], dim=1)  # [num_edges, 2 * node_dim + edge_dim]

        # Apply neural network layer if specified
        if self.use_nn:
            combined_features = self.nnlayer(combined_features)
        
        # Save the combined features as a new edge vector b_ij
        g.edata['b_ij'] = combined_features
        

        # Pool locally across all b_ij with the same starting node i
        if self.node_agg == 'mean':
            g.update_all(fn.copy_edge('b_ij', 'm'), fn.mean('m', 'b_i'))  # Mean pooling
        elif self.node_agg == 'sum':
            g.update_all(fn.copy_edge('b_ij', 'm'), fn.sum('m', 'b_i'))  # Sum pooling
        elif self.node_agg == 'max':
            g.update_all(fn.copy_edge('b_ij', 'm'), fn.max('m', 'b_i'))  # Max pooling
        elif self.node_agg == 'min':
            g.update_all(fn.copy_edge('b_ij', 'm'), fn.min('m', 'b_i'))  # Min pooling
        elif self.node_agg == 'norm':
            g.update_all(fn.copy_edge('b_ij', 'm'), fn.sum('m', 'b_i'))  # Sum pooling
            g.ndata['b_i'] = g.ndata['b_i'] / torch.norm(g.ndata['b_i'], dim=1, keepdim=True)  # Normalize
        else:
            raise NotImplementedError(f"Pooling method '{self.node_agg}' not implemented")


        # Pool across all b_i based on the specified aggregation method
        if self.edge_agg == 'mean':
            pooled_features = g.ndata['b_i'].mean(dim=0, keepdim=True)  # Mean pooling
        elif self.edge_agg == 'sum':
            pooled_features = g.ndata['b_i'].sum(dim=0, keepdim=True)  # Sum pooling
        elif self.edge_agg == 'max':
            pooled_features = g.ndata['b_i'].max(dim=0, keepdim=True).values  # Max pooling
        elif self.edge_agg == 'min':
            pooled_features = g.ndata['b_i'].min(dim=0, keepdim=True).values  # Min pooling
        elif self.edge_agg == 'norm':
            pooled_features = g.ndata['b_i'] / torch.norm(g.ndata['b_i'], dim=1, keepdim=True).sum(dim=0, keepdim=True)  # Normalized pooling
        else:
            raise NotImplementedError(f"Pooling method '{self.edge_agg}' not implemented")
        
        return pooled_features
    

class BondEnvironmentAggregatorSimplified:
    def __init__(self,use_nn=False,node_feat_size: int = 0, edge_feat_size: int = 0):
        if use_nn : self.nnlayer = nn.Linear(2*node_feat_size, edge_feat_size)
        pass

    def forward(self,g):
        # Get node and edge features
        node_features = g.ndata['h']  # Node features [num_nodes, node_dim]
        edge_features = g.edata['e']  # Edge features [num_edges, edge_dim]

        # Combine node and edge features for each edge
        src, dst = g.edges()  # Get source and destination node indices for each edge
        src_node_features = node_features[src]  # Source node features [num_edges, node_dim]
        dst_node_features = node_features[dst]  # Destination node features [num_edges, node_dim]

        # Concatenate source, destination, and edge features
        combined_features = torch.cat([src_node_features, dst_node_features, edge_features], dim=1)  # [num_edges, 2 * node_dim + edge_dim]

        # Apply neural network layer if specified
        if self.use_nn:
            combined_features = self.nnlayer(combined_features)
        
        # Save the combined features as a new edge vector b_ij
        g.edata['b_ij'] = combined_features
        

        # Pool globally across all b_ij
        if self.edge_agg == 'mean':
            pooled_features = g.edata['b_ij'].mean(dim=0, keepdim=True)  # Mean pooling
        elif self.edge_agg == 'sum':
            pooled_features = g.edata['b_ij'].sum(dim=0, keepdim=True)  # Sum pooling
        elif self.edge_agg == 'max':
            pooled_features = g.edata['b_ij'].max(dim=0, keepdim=True).values  # Max pooling
        elif self.edge_agg == 'min':
            pooled_features = g.edata['b_ij'].min(dim=0, keepdim=True).values  # Min pooling
        elif self.edge_agg == 'norm':
            pooled_features = g.edata['b_ij'] / torch.norm(g.edata['b_ij'], dim=1, keepdim=True).sum(dim=0, keepdim=True)  # Normalized pooling
        else:
            raise NotImplementedError(f"Pooling method '{self.edge_agg}' not implemented")
        
        return pooled_features




