import torch
import torch.nn as nn
from bayesian_torch.layers import LinearReparameterization

# Define a Bayesian Neural Network
class BNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=1):
        super().__init__()
        self.fc1 = LinearReparameterization(
            in_features=input_dim,
            out_features=hidden_dim,
            prior_mean=0.0,
            prior_variance=1.0,
        )
        self.fc2 = LinearReparameterization(
            in_features=hidden_dim,
            out_features=output_dim,
            prior_mean=0.0,
            prior_variance=1.0,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.fc1(x)  # Returns (output, KL divergence)
        x = self.relu(x)
        x, _ = self.fc2(x)
        return x

# Training with Uncertainty-Aware Loss
model = BNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    
    # Forward pass (samples weights from variational posterior)
    outputs, kl = model(x_train)
    
    # Compute loss: Negative Log Likelihood + KL Divergence
    nll = torch.nn.functional.mse_loss(outputs, y_train)
    loss = nll + kl / len(x_train)  # Scale KL term
    
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
# Monte Carlo Sampling for Uncertainty
with torch.no_grad():
    predictions = torch.stack([model(x_test) for _ in range(100)])
mean = predictions.mean(dim=0)
std = predictions.std(dim=0)