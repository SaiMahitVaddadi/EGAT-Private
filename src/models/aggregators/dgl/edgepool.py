import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn

class EdgePoolLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads=1):
        super(EdgePoolLayer, self).__init__()
        self.num_heads = num_heads
        self.att_mlp = nn.Linear(2 * node_dim + edge_dim, num_heads)
        self.node_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, edge_dim)

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_feats

            # Edge score with attention: a_ij = MLP([h_i || h_j || e_ij])
            def edge_score(edges):
                h_i = edges.src['h']
                h_j = edges.dst['h']
                e_ij = edges.data['e']
                combined = torch.cat([h_i, h_j, e_ij], dim=-1)
                score = self.att_mlp(combined)  # [E, heads]
                return {'score': score.mean(dim=1)}  # mean across heads → [E]

            g.apply_edges(edge_score)
            edge_scores = g.edata['score']  # [E]

            # Select top-k edges to contract
            num_edges = g.number_of_edges()
            k = max(1, int(0.5 * num_edges))  # contraction ratio (can be a param)
            topk_scores, topk_indices = torch.topk(edge_scores, k)

            edges_to_contract = topk_indices
            src, dst = g.find_edges(edges_to_contract)

            # Create node mapping: each contracted pair → new supernode
            new_node_map = {}
            merged_feats = []
            used = set()

            for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
                if u in used or v in used:
                    continue
                used.update([u, v])
                new_node_id = len(new_node_map)
                new_node_map[u] = new_node_id
                new_node_map[v] = new_node_id

                # Average their features
                new_feat = (node_feats[u] + node_feats[v]) / 2
                merged_feats.append(new_feat)

            # Add remaining unmerged nodes
            for i in range(g.number_of_nodes()):
                if i not in new_node_map:
                    new_node_id = len(new_node_map)
                    new_node_map[i] = new_node_id
                    merged_feats.append(node_feats[i])

            new_num_nodes = len(new_node_map)
            new_feats = torch.stack(merged_feats)

            # Build new graph (connectivity could be redefined based on supernode rels)
            remap = lambda x: new_node_map[x.item()]
            src_all, dst_all = g.edges()
            new_src = []
            new_dst = []
            new_edge_feats = []

            for u, v, e in zip(src_all, dst_all, edge_feats):
                u_new, v_new = remap(u), remap(v)
                if u_new != v_new:  # no self-loops
                    new_src.append(u_new)
                    new_dst.append(v_new)
                    new_edge_feats.append(e)

            new_g = dgl.graph((new_src, new_dst), num_nodes=new_num_nodes, device=node_feats.device)
            new_edge_feats = torch.stack(new_edge_feats) if new_edge_feats else torch.zeros((0, edge_feats.shape[1]), device=node_feats.device)

            return new_g, self.node_proj(new_feats), self.edge_proj(new_edge_feats)


class EdgePool(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers=2, num_heads=4):
        super(EdgePool, self).__init__()
        self.layers = nn.ModuleList([
            EdgePoolLayer(node_dim, edge_dim, num_heads) for _ in range(num_layers)
        ])
        self.node_dim = node_dim

    def forward(self, g, node_feats, edge_feats):
        for layer in self.layers:
            g, node_feats, edge_feats = layer(g, node_feats, edge_feats)

        graph_emb = node_feats.mean(dim=0)  # global readout
        return graph_emb


class EdgePoolLayerwAttention(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(EdgePoolLayer, self).__init__()
        self.node_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, edge_dim)

    def forward(self, g, node_feats, edge_feats, edge_attention):
        # edge_attention: [E] or [E, num_heads]
        if edge_attention.dim() == 2:
            edge_attention = edge_attention.mean(dim=1)

        # Select top edges
        num_edges = g.number_of_edges()
        k = max(1, int(0.5 * num_edges))
        topk_scores, topk_indices = torch.topk(edge_attention, k)

        src, dst = g.find_edges(topk_indices)
        new_node_map = {}
        merged_feats = []
        used = set()

        for u, v in zip(src.tolist(), dst.tolist()):
            if u in used or v in used:
                continue
            used.update([u, v])
            new_id = len(new_node_map)
            new_node_map[u] = new_id
            new_node_map[v] = new_id
            merged_feats.append((node_feats[u] + node_feats[v]) / 2)

        for i in range(g.number_of_nodes()):
            if i not in new_node_map:
                new_node_map[i] = len(new_node_map)
                merged_feats.append(node_feats[i])

        new_feats = torch.stack(merged_feats)
        src_all, dst_all = g.edges()
        remap = lambda x: new_node_map[x.item()]
        new_src, new_dst, new_edge_feats = [], [], []

        for u, v, e in zip(src_all, dst_all, edge_feats):
            u_new, v_new = remap(u), remap(v)
            if u_new != v_new:
                new_src.append(u_new)
                new_dst.append(v_new)
                new_edge_feats.append(e)

        new_g = dgl.graph((new_src, new_dst), num_nodes=len(merged_feats), device=node_feats.device)
        if new_edge_feats:
            new_edge_feats = torch.stack(new_edge_feats)
        else:
            new_edge_feats = torch.zeros((0, edge_feats.shape[1]), device=node_feats.device)

        return new_g, self.node_proj(new_feats), self.edge_proj(new_edge_feats)

class EdgePoolwAttention(nn.Module):
    def __init__(self, in_dim, num_layers=2):
        super(EdgePool, self).__init__()
        self.layers = nn.ModuleList([
            EdgePoolLayerwAttention(in_dim, in_dim) for _ in range(num_layers)
        ])
        self.readout = lambda x: x.mean(dim=0)

    def forward(self, g, node_feats, edge_feats, edge_attentions):
        """
        edge_attentions: List[Tensor] where each tensor is [E, H] or [E]
                         Each entry corresponds to one layer.
        """
        for i, layer in enumerate(self.layers):
            attn = edge_attentions[i]
            g, node_feats, edge_feats = layer(g, node_feats, edge_feats, attn)

        return self.readout(node_feats)
