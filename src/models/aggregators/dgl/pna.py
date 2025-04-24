import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

EPS = 1e-5


def aggregate_mean(h):
    return torch.mean(h, dim=1)


def aggregate_max(h):
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    return torch.min(h, dim=1)[0]


def aggregate_std(h):
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, n=3):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n))
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1. / n)
    return rooted_h_n


def aggregate_moment_3(h):
    return aggregate_moment(h, n=3)


def aggregate_moment_4(h):
    return aggregate_moment(h, n=4)


def aggregate_moment_5(h):
    return aggregate_moment(h, n=5)


def aggregate_sum(h):
    return torch.sum(h, dim=1)


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var, 'moment3': aggregate_moment_3, 'moment4': aggregate_moment_4,
               'moment5': aggregate_moment_5}


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output

def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    return h * (avg_d["log"] / np.log(D + 1))


SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation}




class PNBasedEnvironmentalPooling(nn.Module):

    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, batch_norm, residual,
                posttrans_layers=1):
        """
        A simpler version of PNA layer that simply aggregates the neighbourhood (similar to GCN and GIN),
        without using the pretransformation or the tower mechanisms of the MPNN. It does not support edge features.

        :param in_dim:              size of the input per node
        :param out_dim:             size of the output per node
        :param aggregators:         set of aggregation function identifiers
        :param scalers:             set of scaling functions identifiers
        :param avg_d:               average degree of nodes in the training set, used by scalers to normalize
        :param dropout:             dropout used
        :param batch_norm:          whether to use batch normalisation
        :param posttrans_layers:    number of layers in the transformation after the aggregation
        """
        super().__init__()

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

        self.aggregators = aggregators
        self.scalers = scalers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers)) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu',
                             last_activation='none')
        self.avg_d = avg_d


    def reduce_func(self, nodes):
        h = nodes.mailbox['m']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}

    def edge_reduce_func(self, edge):
        h = edge.src['h']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'m': h}

    def forward(self, g):
        g.ndata['h'] = h

        # aggregation
        g.update_all(fn.copy_u('h', 'm'), self.reduce_func)
        g.update_all(fn.copy_e('h', 'm'), self.edge_reduce_func)
        h = g.ndata['h']
        e = g.edata['e']
        h = torch.cat([h, e], dim=-1)

        # posttransformation
        h = self.posttrans(h)

        # batch normalization and residual
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)