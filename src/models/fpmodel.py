import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_nodes, dropout, activation=None, softmax=None):
        super(FeedforwardNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        if isinstance(dropout, list):
            self.dropout = [nn.Dropout(p) for p in dropout]
        else:
            self.dropout = nn.Dropout(dropout) if dropout is not None else None
        if isinstance(activation, list):
            self.activ = [getattr(nn, act)() for act in activation]
        else:
            self.activ = [getattr(nn, activation)()] if activation is not None else [None] * hidden_layers
        self.smax = nn.softmax(dim=1) if softmax is not None else None

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_nodes[0]))

        # Hidden layers
        for i in range(1, hidden_layers):
            self.layers.append(nn.Linear(hidden_nodes[i-1], hidden_nodes[i]))

        # Output layer
        self.layers.append(nn.Linear(hidden_nodes[-1], output_size))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.activ is not None:
                try:
                    x = self.activ[i](x)
                except:
                    x = self.activ[0](x)
            if self.dropout is not None:
                try:
                    x = self.dropout[i](x)
                except:
                    x = self.dropout[0](x)
        x = self.layers[-1](x)
        if self.smax == 'smax':
            x = self.smax(x)
        elif self.smax == 'act':
            try:
                x = self.activ[i+1](x)
            except:
                x = self.activ[0](x)
        return x