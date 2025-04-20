import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv, GraphConv

class SAGPoolLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, pooling_ratio=0.5, num_heads=1, use_edge_attention=True):
        super(SAGPoolLayer, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.use_edge_attention = use_edge_attention
        self.num_heads = num_heads

        # For node embedding and attention scoring
        self.gat_score = GATConv(in_dim, 1, num_heads=num_heads, allow_zero_in_degree=True)

        # For node feature update
        self.gnn_embed = GraphConv(in_dim, hidden_dim)

        if use_edge_attention:
            self.edge_att_linear = nn.Linear(in_dim * 2 + in_dim, num_heads)  # (h_i, h_j, e_ij) → scores

    def forward(self, g, node_feats, edge_feats):
        # Compute node scores with GAT (multiple heads)
        attn_scores = self.gat_score(g, node_feats).mean(dim=1)  # [N], average across heads

        # Top-k node selection
        k = max(1, int(self.pooling_ratio * node_feats.shape[0]))
        topk_scores, topk_indices = torch.topk(attn_scores, k)

        # Extract the subgraph with top-k nodes
        g_pooled = dgl.node_subgraph(g, topk_indices)

        # Map global to local node indices for edge features
        mask = torch.zeros(node_feats.shape[0], dtype=torch.bool, device=node_feats.device)
        mask[topk_indices] = True

        node_feats_pooled = node_feats[topk_indices]

        # Update node embeddings
        node_feats_pooled = F.relu(self.gnn_embed(g_pooled, node_feats_pooled))

        # Pool edge features using optional attention mechanism
        if self.use_edge_attention:
            with g_pooled.local_scope():
                g_pooled.ndata['h'] = node_feats_pooled
                g_pooled.edata['e'] = edge_feats[g_pooled.edata[dgl.EID]]  # original edge IDs

                def edge_attention(edges):
                    h_i = edges.src['h']
                    h_j = edges.dst['h']
                    e_ij = edges.data['e']
                    x = torch.cat([h_i, h_j, e_ij], dim=-1)
                    scores = self.edge_att_linear(x)  # [E, heads]
                    scores = scores.mean(dim=-1, keepdim=True)
                    return {'e_new': e_ij * scores}

                g_pooled.apply_edges(edge_attention)
                edge_feats_pooled = g_pooled.edata['e_new']
        else:
            edge_feats_pooled = edge_feats[g_pooled.edata[dgl.EID]]

        return g_pooled, node_feats_pooled, edge_feats_pooled
class SAGPool(nn.Module):
    def __init__(self, in_dim, hidden_dim, pooling_ratio=0.5, num_layers=2, num_heads=1, use_edge_attention=True):
        super(SAGPool, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SAGPoolLayer(in_dim, hidden_dim, pooling_ratio, num_heads, use_edge_attention))
            in_dim = hidden_dim

    def forward(self, g, node_feats, edge_feats):
        for layer in self.layers:
            g, node_feats, edge_feats = layer(g, node_feats, edge_feats)

        graph_embedding = node_feats.mean(dim=0)  # Could be sum/max/attention
        return graph_embedding


class SAGPoolLayerwAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, pooling_ratio=0.5):
        super(SAGPoolLayer, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.gnn_embed = GraphConv(in_dim, hidden_dim)

    def forward(self, g, node_feats, edge_feats, edge_attention):
        # edge_attention: [E] or [E, num_heads] → average if needed
        if edge_attention.dim() == 2:
            edge_attention = edge_attention.mean(dim=1)

        # Score nodes using edge attentions: node score = sum of attention to/from node
        g.edata['a'] = edge_attention
        g.update_all(fn.copy_e('a', 'm'), fn.sum('m', 'node_score'))
        node_score = g.ndata['node_score']  # [N]

        k = max(1, int(self.pooling_ratio * g.number_of_nodes()))
        topk_val, topk_idx = torch.topk(node_score, k)

        g_pooled = dgl.node_subgraph(g, topk_idx)
        node_feats_pooled = node_feats[topk_idx]
        node_feats_pooled = F.relu(self.gnn_embed(g_pooled, node_feats_pooled))

        edge_feats_pooled = edge_feats[g_pooled.edata[dgl.EID]]
        return g_pooled, node_feats_pooled, edge_feats_pooled

class SAGPoolwAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, pooling_ratio=0.5, num_layers=2):
        super(SAGPool, self).__init__()
        self.layers = nn.ModuleList([
            SAGPoolLayerwAttention(in_dim if i == 0 else hidden_dim, hidden_dim, pooling_ratio)
            for i in range(num_layers)
        ])
        self.readout = lambda x: x.mean(dim=0)  # could be sum/max/attention

    def forward(self, g, node_feats, edge_feats, edge_attentions):
        """
        edge_attentions: List[Tensor] where each tensor is [E, H] or [E]
                         Each entry corresponds to one layer.
        """
        for i, layer in enumerate(self.layers):
            attn = edge_attentions[i]
            g, node_feats, edge_feats = layer(g, node_feats, edge_feats, attn)

        return self.readout(node_feats)
