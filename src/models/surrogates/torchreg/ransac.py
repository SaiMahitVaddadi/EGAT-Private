class RANSACRegressor(nn.Module):
    def __init__(self, min_samples=None, residual_threshold=None, max_trials=100):
        super().__init__()
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.coef_ = nn.Parameter(None)
        
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        n_samples, n_features = X.shape
        if self.min_samples is None:
            self.min_samples = n_features + 1
            
        if self.residual_threshold is None:
            self.residual_threshold = 1.5 * torch.median(torch.abs(y - torch.median(y)))
            
        best_inliers = None
        best_coef = None
        best_score = -1
        
        for _ in range(self.max_trials):
            # Randomly select samples
            idx = torch.randperm(n_samples)[:self.min_samples]
            X_sub = X[idx]
            y_sub = y[idx]
            
            # Fit model
            try:
                coef = torch.linalg.lstsq(X_sub, y_sub).solution
            except:
                continue
                
            # Compute residuals
            residuals = torch.abs(y - X @ coef)
            inliers = residuals < self.residual_threshold
            
            # Score model
            score = inliers.sum().item()
            
            if score > best_score:
                best_score = score
                best_coef = coef
                best_inliers = inliers
                
        # Refit with all inliers
        if best_inliers is not None and best_inliers.sum() > 0:
            X_inliers = X[best_inliers]
            y_inliers = y[best_inliers]
            best_coef = torch.linalg.lstsq(X_inliers, y_inliers).solution
            
        self.coef_ = nn.Parameter(best_coef if best_coef is not None else torch.zeros(n_features, device=X.device))
    
    def forward(self, X):
        return X @ self.coef_