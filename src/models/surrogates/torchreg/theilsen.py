class TheilSenRegressor(nn.Module):
    def __init__(self, n_subsamples=None, max_iter=1000, tol=1e-3):
        super().__init__()
        self.n_subsamples = n_subsamples
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = nn.Parameter(None)
        
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        n_samples, n_features = X.shape
        if self.n_subsamples is None:
            self.n_subsamples = min(1000, n_samples)
            
        # Generate random subsets
        subsets = torch.randint(0, n_samples, (self.max_iter, self.n_subsamples))
        
        # Compute slopes for each subset
        slopes = []
        for subset in subsets:
            X_sub = X[subset]
            y_sub = y[subset]
            
            # Solve for this subset
            try:
                coef = torch.linalg.lstsq(X_sub, y_sub).solution
                slopes.append(coef)
            except:
                continue
                
        if not slopes:
            raise RuntimeError("No valid subsets found")
            
        slopes = torch.stack(slopes)
        
        # Take median of all slopes
        self.coef_ = nn.Parameter(torch.median(slopes, dim=0).values)
    
    def forward(self, X):
        return X @ self.coef_