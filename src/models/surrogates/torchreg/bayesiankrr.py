import torch
import torch.nn as nn
import torch.distributions as dist

class BayesianRidge(nn.Module):
    def __init__(self, n_features, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6):
        super(BayesianRidge, self).__init__()
        self.w = nn.Parameter(torch.zeros(n_features))
        self.alpha = alpha_1 / alpha_2  # precision of noise
        self.lambda_ = lambda_1 / lambda_2  # precision of weights
        
        # Gamma priors for alpha and lambda
        self.alpha_a = torch.tensor(alpha_1)
        self.alpha_b = torch.tensor(alpha_2)
        self.lambda_a = torch.tensor(lambda_1)
        self.lambda_b = torch.tensor(lambda_2)
        
    def forward(self, X):
        return X @ self.w
    
    def fit(self, X, y, n_iter=300):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        for _ in range(n_iter):
            # Update weights
            A = self.alpha * X.T @ X + self.lambda_ * torch.eye(X.shape[1])
            self.w.data = torch.linalg.solve(A, self.alpha * X.T @ y)
            
            # Update alpha
            residual = y - self.forward(X)
            alpha_a_post = self.alpha_a + X.shape[0] / 2
            alpha_b_post = self.alpha_b + (residual @ residual) / 2
            self.alpha = alpha_a_post / alpha_b_post
            
            # Update lambda
            lambda_a_post = self.lambda_a + X.shape[1] / 2
            lambda_b_post = self.lambda_b + (self.w @ self.w) / 2
            self.lambda_ = lambda_a_post / lambda_b_post
    
    def predict(self, X, return_std=False):
        X = torch.tensor(X, dtype=torch.float32)
        y_pred = self.forward(X)
        if return_std:
            A = self.alpha * X.T @ X + self.lambda_ * torch.eye(X.shape[1])
            cov = torch.linalg.inv(A)
            y_std = torch.sqrt(torch.diag(X @ cov @ X.T) + 1/self.alpha)
            return y_pred.detach().numpy(), y_std.detach().numpy()
        return y_pred.detach().numpy()

