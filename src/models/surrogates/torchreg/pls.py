class PLSRegression(nn.Module):
    def __init__(self, n_components=2, max_iter=500, tol=1e-6):
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.register_buffer('x_weights_', None)
        self.register_buffer('y_weights_', None)
        
    def fit(self, X, Y):
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        
        # Initialize weights
        self.x_weights_ = torch.randn(X.shape[1], self.n_components, device=X.device)
        self.y_weights_ = torch.randn(Y.shape[1], self.n_components, device=X.device)
        
        for _ in range(self.max_iter):
            # X weights
            x_weights_old = self.x_weights_.clone()
            x_scores = X @ self.x_weights_
            self.x_weights_ = X.T @ (Y @ self.y_weights_)
            self.x_weights_ = F.normalize(self.x_weights_, dim=0)
            
            # Y weights
            y_scores = Y @ self.y_weights_
            self.y_weights_ = Y.T @ (X @ self.x_weights_)
            self.y_weights_ = F.normalize(self.y_weights_, dim=0)
            
            # Check convergence
            if torch.norm(self.x_weights_ - x_weights_old) < self.tol:
                break
                
        # Compute regression coefficients
        x_scores = X @ self.x_weights_
        self.coef_ = self.x_weights_ @ torch.pinverse(x_scores.T @ X) @ x_scores.T @ Y
        
    def forward(self, X):
        return X @ self.coef_