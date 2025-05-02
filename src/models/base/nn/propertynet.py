
import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json
from egat import EGATConv,EGATConvResid,EGATConvSA,EGATConvResidSA

class PropertyNet(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=1,activation='GELU',bias=True,smax=None):
        super(PropertyNet, self).__init__()
        self.setuplayers(input_dim, hidden_dim, output_dim,activation,bias,smax)
        
    def setuplayers(self, input_dim=2048, hidden_dim=256, output_dim=1,activation='GELU',bias=True,smax=None):
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias if isinstance(bias, bool) else bias[0])
        self.activationlayers(activation)
        self.fc2 = nn.Linear(hidden_dim, output_dim,bias=bias if isinstance(bias, bool) else bias[1])
        if smax is not None:
            self.softmax = nn.Softmax(dim=smax)
    
    def activationlayers(self,activation='GELU'):
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)() if activation is not None else None
        elif isinstance(activation, list) and len(activation) == 2:
            self.activation = getattr(nn, activation[0])() if activation[0] is not None else None
            self.second_activation = getattr(nn, activation[1])() if activation[1] is not None else None
    
    
    def dropoutlayers(self, dropout=None):
        if isinstance(dropout, str):
            self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        elif isinstance(dropout, list) and len(dropout) == 2:
            self.dropout = nn.Dropout(p=dropout[0]) if dropout[0] is not None else None
            self.second_dropout = nn.Dropout(p=dropout[1]) if dropout[1] is not None else None


    def forward(self, x):
        x = self.fc1(x)
        if hasattr(self, 'activation') and self.activation is not None: x = self.activation(x)
        if hasattr(self, 'dropout') and self.dropout is not None: x = self.dropout(x)
        x = self.fc2(x)
        if hasattr(self, 'second_activation') and self.second_activation is not None: x = self.second_activation(x)
        if hasattr(self, 'second_dropout') and self.second_dropout is not None: x = self.second_dropout(x)
        if hasattr(self, 'softmax'): x = self.softmax(x)
        return x

class PropertyNetNMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=1, activation='GELU', bias=True,layers = 2,smax=None):
        super(PropertyNetNMLP, self).__init__()
        self.setuplayers(input_dim, hidden_dim, output_dim, activation, bias,layers,smax)
        
    def setuphiddendim(self,hidden_dim,layers):
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * layers
        elif isinstance(hidden_dim, list):
            if len(hidden_dim) != layers:
                raise ValueError("Length of hidden_dim list must match the number of layers.")
        return hidden_dim
            
    def setupbias(self,bias,layers):
        if isinstance(bias, bool):
            bias = [bias] * (layers + 1)
        elif isinstance(bias, list):
            if len(bias) != layers + 1:
                raise ValueError("Length of bias list must match the number of layers + 1.")
        return bias
        
    def activationlayers(self,activation='GELU'):
        if isinstance(activation, str):
            activationlayer = getattr(nn, activation)() if activation is not None else None
        elif isinstance(activation, list):
            activationlayer = [getattr(nn, act)() if act is not None else None for act in activation]
        else:
            raise ValueError("activation must be a string or a list of strings.")
        return activationlayer
    
    def dropoutlayers(self, dropout=None):
        if isinstance(dropout, (int, float)):
            dropoutlayer = nn.Dropout(p=dropout) if dropout is not None else None
        elif isinstance(dropout, list):
            dropoutlayer = [nn.Dropout(p=do) if do is not None else None for do in dropout]
        else:
            raise ValueError("dropout must be a float or a list of floats.")
        return dropoutlayer
        
    def setupsmax(self,smax):
        if smax is not None:
            self.softmax = nn.Softmax(dim=smax)
    
    def appendtolayerbasefcn(self,layers,alist,i=0):
        if isinstance(alist, list):
            if alist[i] is not None: layers.append(alist[i])
        else:
            if alist is not None: layers.append(alist)
        return layers

    def createlayer(self,layers,indim,outdim,bias,activation,dropout,i=0):
        layers.append(nn.Linear(indim,outdim, bias=bias))
        layers = self.appendtolayerbasefcn(layers,activation,i)
        layers = self.appendtolayerbasefcn(layers,dropout,i)
        return layers

    def setuplayers(self, input_dim=2048, hidden_dim=256, output_dim=1, activation='GELU', bias=True,layers = 2,smax=None):
        hidden_dim = self.setuphiddendim(hidden_dim,layers)
        bias = self.setupbias(bias,layers)
        activation = self.activationlayers(activation)
        dropout = self.dropoutlayers(dropout)
        self.setupsmax(smax)
        self.layers = []
        dims = [input_dim] + hidden_dim + [output_dim]
        for i in range(layers):
            self.layers = self.createlayer(self.layers,dims[i],dims[i+1],bias[i],activation,dropout,i)
        self.layers.append(self.softmax)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class PropertyNetNMLPAddons(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=1, activation='GELU', bias=True,layers = 2,smax=None,addons=None):
        super(PropertyNetNMLP, self).__init__()
        self.setuplayers(input_dim, hidden_dim, output_dim, activation, bias,layers,smax)
        
    def setuphiddendim(self,hidden_dim,layers):
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * layers
        elif isinstance(hidden_dim, list):
            if len(hidden_dim) != layers:
                raise ValueError("Length of hidden_dim list must match the number of layers.")
        return hidden_dim
            
    def setupbias(self,bias,layers):
        if isinstance(bias, bool):
            bias = [bias] * (layers + 1)
        elif isinstance(bias, list):
            if len(bias) != layers + 1:
                raise ValueError("Length of bias list must match the number of layers + 1.")
        return bias
        
    def activationlayers(self,activation='GELU'):
        if isinstance(activation, str):
            activationlayer = getattr(nn, activation)() if activation is not None else None
        elif isinstance(activation, list):
            activationlayer = [getattr(nn, act)() if act is not None else None for act in activation]
        else:
            raise ValueError("activation must be a string or a list of strings.")
        return activationlayer
    
    def dropoutlayers(self, dropout=None):
        if isinstance(dropout, (int, float)):
            dropoutlayer = nn.Dropout(p=dropout) if dropout is not None else None
        elif isinstance(dropout, list):
            dropoutlayer = [nn.Dropout(p=do) if do is not None else None for do in dropout]
        else:
            raise ValueError("dropout must be a float or a list of floats.")
        return dropoutlayer
        
    def setupsmax(self,smax):
        if smax is not None:
            self.softmax = nn.Softmax(dim=smax)
    
    def appendtolayerbasefcn(self,layers,alist,i=0):
        if isinstance(alist, list):
            if alist[i] is not None: layers.append(alist[i])
        else:
            if alist is not None: layers.append(alist)
        return layers

    def createlayer(self,layers,indim,outdim,bias,activation,dropout,i=0):
        layers.append(nn.Linear(indim,outdim, bias=bias))
        layers = self.appendtolayerbasefcn(layers,activation,i)
        layers = self.appendtolayerbasefcn(layers,dropout,i)
        return layers

    def setuplayers(self, input_dim=2048, hidden_dim=256, output_dim=1, activation='GELU', bias=True,layers = 2,smax=None,addons=None):
        hidden_dim = self.setuphiddendim(hidden_dim,layers)
        bias = self.setupbias(bias,layers)
        activation = self.activationlayers(activation)
        dropout = self.dropoutlayers(dropout)
        self.setupsmax(smax)
        self.layers = []
        dims = [input_dim] + hidden_dim + [output_dim]
        for i in range(layers-1):
            self.layers = self.createlayer(self.layers,dims[i],dims[i+1],bias[i],activation,dropout,i)
        

        self.layers = nn.Sequential(*layers)
        self.lastlayer = [] 
        self.lastlayer = self.createlayer(self.lastlayer,dims[-2]+addons,dims[-1],bias[-1],activation,dropout,layers-1)

        
    def forward(self, x,Hr=None):
        x = self.layers(x)
        if Hr is not None: x = torch.cat((x,Hr),dim=-1)
        x = self.lastlayer(x)
        if hasattr(self, 'softmax'): x = self.softmax(x)
        return x