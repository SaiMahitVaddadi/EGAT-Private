class OrthogonalMatchingPursuit(nn.Module):
    def __init__(self, n_nonzero_coefs=None, tol=None):
        super().__init__()
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.coef_ = nn.Parameter(None)
        
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        n_samples, n_features = X.shape
        if self.n_nonzero_coefs is None and self.tol is None:
            self.n_nonzero_coefs = n_features // 10
            
        residual = y.clone()
        coef = torch.zeros(n_features, device=X.device)
        support = []
        
        for _ in range(n_features if self.tol is not None else self.n_nonzero_coefs):
            # Find best correlated feature
            correlations = torch.abs(X.T @ residual)
            correlations[support] = -1  # Don't select already chosen features
            new_idx = torch.argmax(correlations).item()
            
            # Update support
            support.append(new_idx)
            
            # Solve least squares with current support
            X_support = X[:, support]
            coef_support = torch.linalg.lstsq(X_support, y).solution
            
            # Update residual
            residual = y - X_support @ coef_support
            
            # Check tolerance
            if self.tol is not None and torch.norm(residual) < self.tol:
                break
                
        # Create full coefficient vector
        full_coef = torch.zeros(n_features, device=X.device)
        full_coef[support] = coef_support
        self.coef_ = nn.Parameter(full_coef)
    
    def forward(self, X):
        return X @ self.coef_