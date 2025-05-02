class HuberRegressor(nn.Module):
    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001, tol=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.coef_ = nn.Parameter(None)
        
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        n_samples, n_features = X.shape
        self.coef_ = nn.Parameter(torch.zeros(n_features, device=X.device))
        
        optimizer = torch.optim.LBFGS([self.coef_], line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            residuals = y - X @ self.coef_
            
            # Huber loss
            abs_res = torch.abs(residuals)
            quadratic = torch.min(abs_res, torch.tensor(self.epsilon, device=X.device))
            linear = abs_res - quadratic
            loss = 0.5 * quadratic.pow(2).sum() + self.epsilon * linear.sum()
            
            # Add L2 regularization
            loss += 0.5 * self.alpha * torch.norm(self.coef_, p=2) ** 2
            
            loss.backward()
            return loss
        
        for _ in range(self.max_iter):
            prev_coef = self.coef_.data.clone()
            optimizer.step(closure)
            
            if torch.norm(self.coef_.data - prev_coef) < self.tol:
                break
    
    def forward(self, X):
        return X @ self.coef_