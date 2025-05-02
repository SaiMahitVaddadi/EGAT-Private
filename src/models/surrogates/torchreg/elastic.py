class ElasticNet(nn.Module):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
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
            pred = X @ self.coef_
            mse_loss = 0.5 * torch.mean((pred - y) ** 2)
            
            l1_reg = torch.norm(self.coef_, p=1)
            l2_reg = 0.5 * torch.norm(self.coef_, p=2) ** 2
            
            total_loss = mse_loss + self.alpha * (
                self.l1_ratio * l1_reg + 
                (1 - self.l1_ratio) * l2_reg
            )
            
            total_loss.backward()
            return total_loss
        
        for _ in range(self.max_iter):
            prev_coef = self.coef_.data.clone()
            optimizer.step(closure)
            
            if torch.norm(self.coef_.data - prev_coef) < self.tol:
                break
    
    def forward(self, X):
        return X @ self.coef_