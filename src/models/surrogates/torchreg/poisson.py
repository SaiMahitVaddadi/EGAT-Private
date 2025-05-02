class PoissonRegressor(nn.Module):
    def __init__(self, n_features):
        super(PoissonRegressor, self).__init__()
        self.w = nn.Parameter(torch.zeros(n_features))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, X):
        return torch.exp(X @ self.w + self.bias)
    
    def fit(self, X, y, lr=0.01, n_iter=1000):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for _ in range(n_iter):
            optimizer.zero_grad()
            mu = self.forward(X)
            loss = -torch.sum(y * torch.log(mu) - mu)  # Negative log likelihood
            loss.backward()
            optimizer.step()
    
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.forward(X).detach().numpy()