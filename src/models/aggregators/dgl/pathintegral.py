import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class PIPooling(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=4, edge_feats=0, k=3):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.edge_feats = edge_feats
        self.k = k  # Maximum path length

        # Attention computation layers (only used if not using precomputed scores)
        self.attn_layers = nn.ModuleList()
        for _ in range(num_heads):
            input_dim = 2 * in_feats + edge_feats
            self.attn_layers.append(nn.Linear(input_dim, 1))

        # Learnable path coefficients
        self.alphas = nn.Parameter(torch.Tensor(num_heads, k))
        nn.init.normal_(self.alphas)

        # Projection layer
        self.proj = nn.Linear(num_heads * in_feats, out_feats)

    def forward(self, graph, h, e=None, attn_key=None):
        """Forward pass with attention score handling
        Args:
            graph: DGL graph
            h: Node features (N, in_feats)
            e: Edge features (E, edge_feats) - only needed if computing new attention scores
            attn_key: Key for precomputed attention scores in graph.edata (E, num_heads)
        """
        with graph.local_scope():
            N = graph.number_of_nodes()
            E = graph.number_of_edges()
            
            # Get or compute attention scores
            if attn_key is not None:
                # Use precomputed attention scores from graph
                assert attn_key in graph.edata, f"Key '{attn_key}' not found in graph edge data"
                attn_scores = graph.edata[attn_key]
                assert attn_scores.shape == (E, self.num_heads), \
                    f"Invalid attention scores shape {attn_scores.shape}, expected {(E, self.num_heads)}"
            else:
                # Compute new attention scores dynamically
                src, dst = graph.edges()
                h_src = h[src]
                h_dst = h[dst]
                
                attn_scores = []
                for m in range(self.num_heads):
                    # Combine source, destination, and edge features
                    if e is not None:
                        combined = torch.cat([h_src, h_dst, e], dim=1)
                    else:
                        combined = torch.cat([h_src, h_dst], dim=1)
                    
                    # Compute attention for this head
                    a_m = self.attn_layers[m](combined)
                    a_m = F.leaky_relu(a_m).squeeze(1)
                    attn_scores.append(a_m)
                
                attn_scores = torch.stack(attn_scores, dim=1)  # (E, num_heads)

            # Path integral accumulation
            outputs = []
            for m in range(self.num_heads):
                graph.edata['a_m'] = attn_scores[:, m]
                H_total = torch.zeros(N, self.in_feats, device=h.device)
                H_current = h.clone()

                for l in range(self.k):
                    # Perform message passing step
                    graph.ndata['h'] = H_current
                    graph.update_all(fn.u_mul_e('h', 'a_m', 'm'), fn.sum('m', 'h_next'))
                    H_step = graph.ndata.pop('h_next')

                    # Accumulate with learned coefficient
                    H_total += self.alphas[m, l] * H_step
                    H_current = H_step

                outputs.append(H_total)

            # Combine heads and project
            h_out = torch.cat(outputs, dim=1)
            return self.proj(h_out)