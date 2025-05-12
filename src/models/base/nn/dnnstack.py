import torch
from torch import nn
from .propertynet import PropertyNet,PropertyNetNMLP,PropertyNetNMLPAddons


class BaseNNCommands(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.params = cfg
    
    def hiddenlayerpropnet(self):
        if self.params.AggregateReaction == 'Concat':
            if not self.params.MixingLayer:
                hidden_dim = self.params.hidden_dim*self.params.num_heads*4
            else:
                hidden_dim = self.params.hidden_dim*self.params.num_heads*2
        else:
            hidden_dim = self.params.hidden_dim*self.params.num_heads*2
        return hidden_dim

    def addons(self,addonlength=0):
        addon = len(self.params.additional) if self.params.additional is not None else 0
        if self.params.addons is not None: addon += addonlength 
        return addon
    
    def propnetoutput(self):
        if self.params.model_type != 'BEP':
            output = len(self.params.target)
        elif self.params.model_type == 'Direct':
            output = 1
        else:
            output = 2
        return output
    
    def getactivation(self,activation='GELU'):
        try:
            activation = getattr(nn, activation)()
        except:
            activation = getattr(nn, 'GELU')()

    def _defaultdims(self,indim=None,hidden_dim=None,outdim=None):
        if indim is None:
            indim = self.hiddenlayerpropnet()
        if outdim is None:
            outdim = self.propnetoutput()
        if hidden_dim is None:
            hidden_dim = self.params.NN_hidden_dim
        return indim,hidden_dim,outdim
    
    def createcustomsoftmax(self,smax='softmax'):
        smax = getattr(nn, smax)()
        self.softmax = smax
    
    def createcustomendconvolution(self,endconv='Linear'):
        endconv = getattr(nn, endconv)()
        self.endconv = endconv

class DeepNeuralNetStack(BaseNNCommands):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.params = cfg
        
    
    def create1mlp(self,addons=0):
        hidden_dim = self.hiddenlayerpropnet()
        output = self.propnetoutput()
        addon = self.addons(addons)
        self.subnets = nn.ModuleList([PropertyNet(hidden_dim + addon, self.params.propnet_hdim, 1,activation=self.params.activation,bias=self.params.Bias,smax=self.params.smax) for _ in range(output)])

    def createsubnetmodel(self,addons=0):
        hidden_dim = self.hiddenlayerpropnet()
        output = self.propnetoutput()
        addon = self.addons(addons)
        self.subnets = nn.ModuleList([PropertyNetNMLP(hidden_dim + addon, self.params.propnet_hdim, 1,activation=self.params.activation,bias=self.params.Bias,smax=self.params.smax,layers = self.params.propnet_layers) for _ in range(output)])

    def createsubnetmodeladdons(self,addons=0):
        hidden_dim = self.hiddenlayerpropnet()
        output = self.propnetoutput()
        addon = self.addons(addons)
        self.subnets = nn.ModuleList([PropertyNetNMLPAddons(hidden_dim, self.params.propnet_hdim, 1,activation=self.params.activation,bias=self.params.Bias,smax=self.params.smax,layers = self.params.propnet_layers,addons=addon) for _ in range(output)])

    
    def create3mlp(self,activation,addons=0):
        hidden_dim = self.hiddenlayerpropnet()
        output = self.propnetoutput()
        activation = self.getactivation(activation)
        addons = self.addons(addons)
        self.mlp1 = nn.Sequential(nn.Linear(hidden_dim+addons, 256, bias=True),activation)
        self.mlp2 = nn.Sequential(nn.Linear(256, 128, bias=True),activation)
        if self.params.model_type == 'Hr2': self.mlp3 = nn.Linear(128 + addons, output, bias=True)
        else: self.mlp3 = nn.Linear(128, output, bias=True)
    
    

    def createcustomMLP(self,activation='GELU',indim=None,outdim=None,dropout=None,bias=None,addons=0):
        indim,_,outdim = self._defaultdims(indim=indim,outdim=outdim)
        addons = self.addons(addons)
        layers = [nn.Linear(indim+addons, outdim, bias=bias)]
        if activation is not None: 
            activation = self.getactivation(activation)
            layers.append(activation)
        if dropout is not None: layers.append(nn.Dropout(p=dropout))
        mlp = nn.Sequential(*layers)
        return mlp,layers
    
    #Check if hidden_dim, activation,dropouts ares list or strings and write the code accordingly
    def createcustom_NMLP(self,activation='GELU',layers=3,indim=None,hidden_dim=None,outdim=None,dropout=None,cascading=False,softmax=None,endconvolution=None,addons=0,bias=None):
        indim,hidden_dim,outdim = self._defaultdims(indim=indim,hidden_dim=hidden_dim,outdim=outdim)
        addons = self.addons(addons)
        if isinstance(hidden_dim, int):
            if cascading:
                hidden_dim = [hidden_dim // (2 ** i) for i in range(layers)]
            else:
                hidden_dim = [hidden_dim] * layers
        if isinstance(activation, str):
            activation = [getattr(nn, activation)() for _ in range(layers)]
        if isinstance(dropout, float):
            dropout = [nn.Dropout(p=dropout) for _ in range(layers)]
        if isinstance(bias, bool):  
            bias = [bias] * (layers + 1)

        dims = [indim] + hidden_dim + [outdim]
        if self.params.model_type == 'Hr': self.mlp1,_ = self.createcustomMLP(activation=activation[0],indim=dims[0] + addons,outdim=dims[1],dropout=dropout[0],bias=bias[0])
        else: self.mlp1,_ = self.createcustomMLP(activation=activation[0],indim=dims[0],outdim=dims[1],dropout=dropout[0],bias=bias[0])
        self.mlp2 = []
        for i in range(1, layers-1):
            _,layer = self.createcustomMLP(activation=activation[i],indim=dims[i],outdim=dims[i+1],dropout=dropout[i],bias=bias[i])
            self.mlp2 += layer
        self.mlp2 = nn.Sequential(*self.mlp2)
        if self.params.model_type == 'Hr2': self.mlp3,_ = self.createcustomMLP(activation=activation[-1],indim=dims[-2] + addons,outdim=dims[-1],dropout=dropout[-1],bias=bias[-1])
        else: self.mlp3,_ = self.createcustomMLP(activation=activation[-1],indim=dims[-2],outdim=dims[-1],dropout=dropout[-1],bias=bias[-1])

        self.createcustomsoftmax(softmax)
        self.createcustomendconvolution(endconvolution)
    
    

    def runsubnet(self,x):
        x = [subnet(x) for subnet in self.subnets]
        x = torch.cat(x, dim=1)
        return x
    
    def runsubnetwithaddons(self,x,addons=None):
        x = [subnet(x,addons) for subnet in self.subnets]
        x = torch.cat(x, dim=1)
        return x

    def runnmlp(self,x,Hr=None):
        if self.params.model_type == 'Hr': x = torch.cat((x,Hr), axis=1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        if self.params.model_type == 'Hr2': x = torch.cat((x,Hr), axis=1)
        x = self.mlp3(x)
        if hasattr(self, 'softmax'): x = self.softmax(x)
        if hasattr(self, 'endconv'): x = self.endconv(x)
        return x
