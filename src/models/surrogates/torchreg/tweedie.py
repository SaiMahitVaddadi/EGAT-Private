class TweedieRegressor(nn.Module):
    def __init__(self, n_features, power=1.5):
        super(TweedieRegressor, self).__init__()
        assert 1 <= power < 2, "Power must be in [1, 2)"
        self.w = nn.Parameter(torch.zeros(n_features))
        self.bias = nn.Parameter(torch.zeros(1))
        self.power = power
        
    def forward(self, X):
        linear = X @ self.w + self.bias
        if self.power == 1:  # Poisson case
            return torch.exp(linear)
        elif self.power == 2:  # Gamma case
            return torch.clamp(linear, min=1e-8)  # prevent negative values
        else:  # Compound Poisson-Gamma
            return torch.pow(torch.clamp(linear, min=1e-8), 1/(2 - self.power))
    
    def fit(self, X, y, lr=0.01, n_iter=1000):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for _ in range(n_iter):
            optimizer.zero_grad()
            mu = self.forward(X)
            
            if self.power == 1:  # Poisson
                loss = -torch.sum(y * torch.log(mu) - mu)
            elif self.power == 2:  # Gamma
                loss = torch.sum(y / mu + torch.log(mu))
            else:  # Tweedie
                loss = torch.sum(
                    -y * torch.pow(mu, 1 - self.power) / (1 - self.power) + 
                    torch.pow(mu, 2 - self.power) / (2 - self.power)
                )
                
            loss.backward()
            optimizer.step()
    
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.forward(X).detach().numpy()