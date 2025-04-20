
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

class DiffPoolLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, assign_dim):
        super(DiffPoolLayer, self).__init__()
        self.embed_gnn = GraphConv(in_dim, hidden_dim)
        self.assign_gnn = GraphConv(in_dim, assign_dim)
        self.linear_embed = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.embed_gnn.weight)
        nn.init.zeros_(self.embed_gnn.bias)
        nn.init.xavier_uniform_(self.assign_gnn.weight)
        nn.init.zeros_(self.assign_gnn.bias)
        nn.init.xavier_uniform_(self.linear_embed.weight)
        nn.init.zeros_(self.linear_embed.bias)

    def forward(self, g, node_feats, edge_feats):
        # g: DGLGraph
        # node_feats: [N, F]
        # edge_feats: [E, D] (optional)

        # Compute embeddings and assignment scores
        Z = F.relu(self.embed_gnn(g, node_feats))         # Node embeddings [N, H]
        S = F.softmax(self.assign_gnn(g, node_feats), dim=-1)  # Assignment scores [N, C]

        # Pooled node features: S^T Z
        X_pooled = torch.matmul(S.transpose(0, 1), Z)  # [C, H]

        # Optional: edge pooling using message passing (simplified)
        with g.local_scope():
            g.ndata['s'] = S  # [N, C]
            g.edata['e'] = edge_feats

            # Use softmax assignment to pool edges (mean pooling as a proxy)
            def edge_pool(edges):
                s_i = edges.src['s']  # [E, C]
                s_j = edges.dst['s']  # [E, C]
                edge_feat = edges.data['e']  # [E, D]

                pooled_edge = torch.matmul(s_i.T, edge_feat) @ s_j  # [C, D] x [C] → [C, C]
                return {'pooled': pooled_edge}

            g.apply_edges(edge_pool)
            edge_pooled = g.edata['pooled'].mean(dim=0)  # [C, C], rough representation

        # Create new graph (pooled)
        batch_size = 1  # assumes single graph
        new_g = dgl.graph(([], []), num_nodes=S.shape[1]).to(node_feats.device)

        return new_g, X_pooled, S, edge_pooled


class DiffPool(nn.Module):
    def __init__(self, in_dim, hidden_dim, assign_dim, num_layers=2):
        super(DiffPool, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        for _ in range(num_layers):
            self.layers.append(DiffPoolLayer(in_dim, hidden_dim, assign_dim))
            in_dim = hidden_dim  # update for next layer

    def forward(self, g, node_feats, edge_feats):
        for i, layer in enumerate(self.layers):
            g, node_feats, S, edge_feats = layer(g, node_feats, edge_feats)

        graph_embedding = node_feats.mean(dim=0)  # Or max/sum/attention
        return graph_embedding

