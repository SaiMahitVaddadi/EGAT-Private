import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv
class BondAggregator:
    def __init__(self,node_feat_size: int = 0, edge_feat_size: int = 0,node_agg='mean',edge_agg='mean',use_nn=False):
        self.use_nn = use_nn
        self.node_agg = node_agg
        self.edge_agg = edge_agg
        if use_nn : self.nnlayer = nn.Linear(2*node_feat_size+edge_feat_size, edge_feat_size)
        pass

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        if self.use_nn:
            nn.init.xavier_uniform_(self.nnlayer.weight)
            nn.init.zeros_(self.nnlayer.bias)


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
    def __init__(self,use_nn=False,node_feat_size: int = 0, edge_feat_size: int = 0,node_agg='mean',edge_agg='mean'):
        self.use_nn = use_nn
        self.node_agg = node_agg
        self.edge_agg = edge_agg
        if use_nn : self.nnlayer = nn.Linear(2*node_feat_size+edge_feat_size, edge_feat_size)
        

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        if self.use_nn:
            nn.init.xavier_uniform_(self.nnlayer.weight)
            nn.init.zeros_(self.nnlayer.bias)

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
            g.update_all(fn.copy_e('b_ij', 'm'), fn.mean('m', 'b_i'))  # Mean pooling
        elif self.node_agg == 'sum':
            g.update_all(fn.copy_e('b_ij', 'm'), fn.sum('m', 'b_i'))  # Sum pooling
        elif self.node_agg == 'max':
            g.update_all(fn.copy_e('b_ij', 'm'), fn.max('m', 'b_i'))  # Max pooling
        elif self.node_agg == 'min':
            g.update_all(fn.copy_e('b_ij', 'm'), fn.min('m', 'b_i'))  # Min pooling
        elif self.node_agg == 'norm':
            g.update_all(fn.copy_e('b_ij', 'm'), fn.sum('m', 'b_i'))  # Sum pooling
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
    def __init__(self,use_nn=False,node_feat_size: int = 0, edge_feat_size: int = 0,node_agg='mean',edge_agg='mean'):
        self.use_nn = use_nn
        self.node_agg = node_agg
        self.edge_agg = edge_agg

        if use_nn : self.nnlayer = nn.Linear(2*node_feat_size+edge_feat_size, edge_feat_size)
        

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        if self.use_nn:
            nn.init.xavier_uniform_(self.nnlayer.weight)
            nn.init.zeros_(self.nnlayer.bias)


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

if __name__ == '__main__':
    # Create a sample graph
    graph = dgl.graph(([0, 1, 2], [1, 2, 3]))
    graph.ndata['h'] = torch.randn(4, 8)  # Node features
    graph.edata['e'] = torch.randn(3, 4)  # Edge features
    graph.edata['a'] = torch.randn(3, 1)  # Edge attention scores
    
    # Initialize the aggregator
    aggregator = BondAggregator(node_feat_size=8, edge_feat_size=4)
    aggregator.reset_parameters()
    # Forward pass
    pooled_feature = aggregator.forward(graph)
    print(pooled_feature)

    # Initialize the aggregator
    aggregator = BondAggregator(node_feat_size=8, edge_feat_size=4,use_nn=True)
    aggregator.reset_parameters()
    # Forward pass
    pooled_feature = aggregator.forward(graph)
    print(pooled_feature)

    # Initialize the aggregator
    aggregator = BondEnvironmentAggregator(node_feat_size=8, edge_feat_size=4)
    aggregator.reset_parameters()
    # Forward pass
    pooled_feature = aggregator.forward(graph)
    print(pooled_feature)

    # Initialize the aggregator
    aggregator = BondEnvironmentAggregator(node_feat_size=8, edge_feat_size=4,use_nn=True)
    aggregator.reset_parameters()
    # Forward pass
    pooled_feature = aggregator.forward(graph)
    print(pooled_feature)

    # Initialize the aggregator
    aggregator = BondEnvironmentAggregatorSimplified(node_feat_size=8, edge_feat_size=4)
    aggregator.reset_parameters()
    # Forward pass
    pooled_feature = aggregator.forward(graph)
    print(pooled_feature)

    # Initialize the aggregator
    aggregator = BondEnvironmentAggregatorSimplified(node_feat_size=8, edge_feat_size=4,use_nn=True)
    aggregator.reset_parameters()
    # Forward pass
    pooled_feature = aggregator.forward(graph)
    print(pooled_feature)
    