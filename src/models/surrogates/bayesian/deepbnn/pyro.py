import torch
import torch.nn as nn
import models.surrogates.bayesian.deepbnn.pyro as pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

# Define a Bayesian Linear Layer with Pyro
class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define weight and bias priors
        self.weight = PyroSample(
            dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.bias = PyroSample(
            dist.Normal(0., 1.).expand([out_features]).to_event(1))

    def forward(self, x):
        return x @ self.weight.T + self.bias

# Define a Bayesian Nonlinear Layer with Pyro
class BayesianNonlinear(PyroModule):
    def __init__(self, in_features, out_features, nonlinearity=nn.ReLU):
        super().__init__()
        self.linear = BayesianLinear(in_features, out_features)
        self.nonlinearity = nonlinearity()

    def forward(self, x):
        x = self.linear(x)
        return self.nonlinearity(x)


# Example Usage in a Model
class BNNLinear(PyroModule):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=1):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class BNNNonlinear(PyroModule):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=1, nonlinearity=nn.ReLU):
        super().__init__()
        self.fc1 = BayesianNonlinear(input_dim, hidden_dim, nonlinearity)
        self.fc2 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)

