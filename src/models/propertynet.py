
import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json
from egat import EGATConv,EGATConvResid,EGATConvSA,EGATConvResidSA

class PropertyNet(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=1,activation='GELU',bias=True):
        super(PropertyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias if isinstance(bias, bool) else bias[0])
        self.activation = getattr(nn, activation)() if activation is not None else None
        self.fc2 = nn.Linear(hidden_dim, output_dim,bias=bias if isinstance(bias, bool) else bias[1])
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class PropertyNetNMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=1, activation='GELU', bias=True,layers = 2):
        super(PropertyNetNMLP, self).__init__()
        
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * layers
        if isinstance(activation, str):
            activation = [activation] * len(hidden_dim)
        if isinstance(bias, bool):
            bias = [bias] * (len(hidden_dim) + 1)
        
        layers = []
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dim):
            layers.append(nn.Linear(in_dim, h_dim, bias=bias[i]))
            layers.append(getattr(nn, activation[i])())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim, bias=bias[-1]))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
