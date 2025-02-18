
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
        self.activation = getattr(nn, activation)()
        self.fc2 = nn.Linear(hidden_dim, output_dim,bias=bias if isinstance(bias, bool) else bias[1])
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x