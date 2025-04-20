import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class ClusterPooling(nn.Module):
    def __init__(
        self,
        in_feats,                  # Input node feature dimension
        edge_feats=None,           # Edge feature dimension (if not using attention)
        num_heads=None,            # Number of attention heads (if using attention)
        use_attention=False,       # If True, use precomputed attention scores
        attn_key='a',              # Key for attention scores in graph.edata
        hard_clustering=False,     # If True, use hard clustering (connected components)
        pooling_method='sum',      # 'sum', 'mean', or 'max' for feature aggregation
        threshold=0.5,             # Edge score threshold for hard clustering
        gnn_cluster=False,         # If True, use GNN for soft clustering
        graph_pooling='mean',      # 'sum', 'mean', 'max' for graph-level embedding
    ):
        super().__init__()
        
        # --- Configuration ---
        self.use_attention = use_attention
        self.hard_clustering = hard_clustering
        self.pooling_method = pooling_method
        self.threshold = threshold
        self.gnn_cluster = gnn_cluster
        self.attn_key = attn_key
        self.graph_pooling = graph_pooling

        # --- Edge Score Computation ---
        if not self.use_attention:
            assert edge_feats is not None, "Edge features must be provided if not using attention"
            self.edge_score_layer = nn.Linear(edge_feats, 1)
        else:
            assert num_heads is not None, "Number of heads must be provided if using attention"
            self.num_heads = num_heads

        # --- Cluster Assignment ---
        if self.gnn_cluster:
            self.assign_gnn = GraphConv(in_feats, in_feats)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights for all learnable layers."""
        if hasattr(self, 'edge_score_layer'):
            nn.init.xavier_uniform_(self.edge_score_layer.weight)
            nn.init.zeros_(self.edge_score_layer.bias)
        if hasattr(self, 'assign_gnn'):
            self.assign_gnn.reset_parameters()

    def forward(self, graph, node_feats, edge_feats=None):
        # === Step 1: Compute Edge Scores ===
        if self.use_attention:
            edge_scores = graph.edata[self.attn_key].mean(dim=1)  # [M]
        else:
            edge_scores = torch.sigmoid(self.edge_score_layer(edge_feats)).squeeze()  # [M]

        # === Step 2: Cluster Assignment ===
        if self.hard_clustering:
            # Hard clustering (connected components)
            subgraph = dgl.edge_subgraph(graph, edge_scores > self.threshold, preserve_nodes=True)
            clusters = dgl.connected_components(subgraph)  # [N]
            num_clusters = clusters.max().item() + 1
        else:
            # Soft clustering (GNN or edge-score-based)
            if self.gnn_cluster:
                assignments = F.softmax(self.assign_gnn(graph, node_feats), dim=-1)  # [N, N']
                num_clusters = assignments.size(1)
            else:
                # Fallback: Assign to highest-scoring neighbor
                adj = graph.adjacency_matrix().to_dense() * edge_scores.unsqueeze(1)
                _, clusters = torch.max(adj, dim=1)  # [N]
                num_clusters = clusters.max().item() + 1

        # === Step 3: Pool Node Features ===
        if self.hard_clustering or not self.gnn_cluster:
            # Hard or simplified soft pooling
            pooled_feats = torch.zeros(num_clusters, node_feats.size(1), device=node_feats.device)
            if self.pooling_method == 'sum':
                pooled_feats.scatter_add_(0, clusters.unsqueeze(1).expand(-1, node_feats.size(1)), node_feats)
            elif self.pooling_method == 'mean':
                pooled_feats.scatter_reduce_(0, clusters.unsqueeze(1).expand(-1, node_feats.size(1)), node_feats, reduce='mean')
            elif self.pooling_method == 'max':
                pooled_feats.scatter_reduce_(0, clusters.unsqueeze(1).expand(-1, node_feats.size(1)), node_feats, reduce='amax')
        else:
            # Soft pooling (GNN-based)
            pooled_feats = torch.matmul(assignments.t(), node_feats)  # [N', d]

        # === Step 4: Build Coarsened Graph ===
        new_graph = dgl.DGLGraph()
        new_graph.add_nodes(num_clusters)
        
        if self.hard_clustering:
            # Hard edge pooling
            src, dst = graph.edges()
            cluster_src = clusters[src]
            cluster_dst = clusters[dst]
            mask = cluster_src != cluster_dst  # Remove self-loops
            new_graph.add_edges(cluster_src[mask], cluster_dst[mask])
        else:
            # Soft edge pooling
            adj = graph.adjacency_matrix().to_dense() * edge_scores.unsqueeze(1)
            if self.gnn_cluster:
                pooled_adj = torch.matmul(assignments.t(), torch.matmul(adj, assignments))
            else:
                pooled_adj = torch.zeros(num_clusters, num_clusters, device=node_feats.device)
                pooled_adj.scatter_add_(0, clusters.unsqueeze(1), torch.matmul(adj, clusters.unsqueeze(1)))
            src, dst = torch.nonzero(pooled_adj > self.threshold, as_tuple=True)
            new_graph.add_edges(src, dst)

        # === Step 5: Graph-Level Embedding ===
        if self.graph_pooling == 'sum':
            graph_embedding = torch.sum(pooled_feats, dim=0)  # [d]
        elif self.graph_pooling == 'mean':
            graph_embedding = torch.mean(pooled_feats, dim=0)  # [d]
        elif self.graph_pooling == 'max':
            graph_embedding = torch.max(pooled_feats, dim=0)[0]  # [d]
        elif self.graph_pooling == 'weighted_sum':
            graph_embedding = torch.matmul(pooled_feats.t(), edge_scores)
        elif self.graph_pooling == 'weighted_mean':
            graph_embedding = torch.matmul(pooled_feats.t(), edge_scores) / edge_scores.sum()
        elif self.graph_pooling == 'normalized':
            graph_embedding = (pooled_feats / torch.norm(pooled_feats, dim=1)).sum(dim=0)

        return new_graph, pooled_feats, graph_embedding