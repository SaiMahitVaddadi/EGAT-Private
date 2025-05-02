import torch
import torch.nn as nn
import torch.optim as optim
from pykan import KolmogorovArnoldNetwork


class FineTuningModelKAN:
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.model = KolmogorovArnoldNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

    def fit(self, X, y, epochs=100, lr=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            predictions = self.model(torch.tensor(X, dtype=torch.float32))
            loss = criterion(predictions, torch.tensor(y, dtype=torch.float32))
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy()
