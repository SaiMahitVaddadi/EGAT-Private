import torch
import torch.nn as nn
import pyro
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

# Example Usage in a Model
class BNN(PyroModule):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=1):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Define Guide (Variational Distribution)
def guide(x, y):
    pyro.module("bnn", model)  # Automatically register variational parameters

# Training with SVI
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

model = BNN()
optimizer = Adam({"lr": 0.01})
svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

# Training loop
for epoch in range(1000):
    loss = svi.step(x_train, y_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")