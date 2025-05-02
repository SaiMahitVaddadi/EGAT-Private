import torch
import torch.nn as nn
from neurobayes.layers import PBNNLinear
from neurobayes.utils import init_random_seed

class TorchFriendlyPBNN(nn.Module):
    def __init__(self, in_features, out_features, rank=1, prior_var=1.0, init_std=0.1, bias=True):
        super().__init__()
        init_random_seed()  # ensure randomness across experiments
        self.pbnn_layer = PBNNLinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            prior_var=prior_var,
            init_std=init_std,
            bias=bias
        )

    def forward(self, x):
        return self.pbnn_layer(x)
