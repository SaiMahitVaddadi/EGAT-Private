class PassiveAggressiveRegressor(nn.Module):
    def __init__(self, C=1.0, epsilon=0.1, max_iter=1000, tol=1e-3):
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = nn.Parameter(None)
        
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        n_samples, n_features = X.shape
        self.coef_ = nn.Parameter(torch.zeros(n_features, device=X.device))
        
        for _ in range(self.max_iter):
            pred = X @ self.coef_
            errors = y - pred
            absolute_errors = torch.abs(errors)
            
            # Find samples where error > epsilon
            update_mask = absolute_errors > self.epsilon
            
            if not update_mask.any():
                break
                
            # Compute update steps
            X_update = X[update_mask]
            y_update = y[update_mask]
            errors_update = errors[update_mask]
            
            # Compute step sizes
            l2_norm = torch.sum(X_update ** 2, dim=1)
            step_sizes = errors_update / (l2_norm + 1e-10)
            step_sizes = torch.clamp(step_sizes, -self.C, self.C)
            
            # Update weights
            updates = (step_sizes.unsqueeze(1) * X_update).mean(dim=0)
            self.coef_.data += updates
            
            # Check convergence
            if torch.norm(updates) < self.tol:
                break
    
    def forward(self, X):
        return X @ self.coef_