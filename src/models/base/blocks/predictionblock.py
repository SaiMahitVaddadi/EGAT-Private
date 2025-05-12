from ..nn.dnnstack import DeepNeuralNetStack
import torch
from dataclasses import dataclass

'''
- Predictor Stack.
    - Add all the new surrogate models and prediction stacks.
'''

@dataclass
class PredictionParams:
    architecture: str  # '1MLP', '3MLP', 'custom', 'subnet'
    model_type: str  # 'Hr', 'Hr2', or other types
    activation: str = None  # Activation function for MLPs
    dropout: float = 0.0  # Dropout rate for custom MLP
    cascading: bool = False  # Whether cascading is enabled
    smax: bool = False  # Whether softmax is applied
    endconvolution: bool = False  # Whether end convolution is applied
    Bias: bool = True  # Whether bias is used in layers


class PredictionBlock(DeepNeuralNetStack):
    def __init__(self, cfg,addonlength=0):
        super().__init__(cfg)
        self.params = cfg
        self.SetupPredictor(addons=addonlength)
    
    def Initialize1MLP(self,addons=0):
        self.create1mlp(addons=addons)

    def Initialize3MLP(self,addons=0):
        self.create3mlp(self.params.activation,addons=addons)

    def InitializeCustomMLP(self):
        self.createcustomMLP(activation=self.params.activation,dropout=self.params.dropout,cascading=self.params.cascading,softmax=self.params.smax,endconvolution=self.params.endconvolution,bias =self.params.Bias)

    def InitializeSubNet(self,addons=0):
        if self.params.model_type == 'Hr2':
            self.createsubnetmodeladdons(addons=addons)
        else:
            self.createsubnetmodel(addons=addons)

    def SetupPredictor(self,addons=0):
        if self.params.architecture == '1MLP':
            self.Initialize1MLP(addons=addons)
        elif self.params.architecture == '3MLP':
            self.Initialize3MLP(addons=addons)
        elif self.params.architecture == 'custom':
            self.InitializeCustomMLP()
        elif self.params.architecture == 'subnet':
            self.InitializeSubNet(addons=addons)
    
    
    def RunSubnet(self,G_features,Hr=None):
        if self.params.model_type == 'Hr': G_features = torch.cat((G_features,Hr), axis=1)
        if self.params.model_type == 'Hr2': x = self.runsubnetwithaddons(G_features,addons=Hr)
        else: x = self.runsubnet(G_features)
        return x 
    
    def RunMLP(self,G_features,Hr=None):
        return self.runnmlp(G_features,Hr=Hr)

    def RunPrediction(self,G_features,Hr=None):
        if self.params.architecture in ['subnet','1MLP']: return self.RunSubnet(G_features,Hr=Hr)
        else: return self.RunMLP(G_features,Hr=Hr)