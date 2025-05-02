class KernelRidge(nn.Module):
    def __init__(self, alpha=1.0, kernel='rbf', gamma=None, degree=3):
        super().__init__()
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.register_buffer('dual_coef_', None)
        self.register_buffer('X_fit_', None)
        
    def _kernel(self, X, Y=None):
        if Y is None:
            Y = X
            
        if self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'poly':
            return (self.gamma * (X @ Y.T) + 1) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]
            dist = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(-1)
            return torch.exp(-self.gamma * dist)
        else:
            raise ValueError("Unknown kernel")
    
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        self.X_fit_ = X.clone()
        K = self._kernel(X)
        n_samples = X.shape[0]
        
        # Solve dual problem
        self.dual_coef_ = torch.linalg.solve(
            K + self.alpha * torch.eye(n_samples, device=X.device), 
            y
        )
    
    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        K = self._kernel(X, self.X_fit_)
        return K @ self.dual_coef_