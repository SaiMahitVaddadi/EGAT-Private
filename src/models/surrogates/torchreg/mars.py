import torch
import torch.nn as nn
import numpy as np
from itertools import combinations

class MARSLayer(nn.Module):
    def __init__(self, 
                 n_features, 
                 max_terms=20, 
                 max_degree=2, 
                 threshold=0.01,
                 pairwise_terms=True):
        """
        MARS (Multivariate Adaptive Regression Splines) Layer
        
        Args:
            n_features (int): Number of input features
            max_terms (int): Maximum number of basis functions (including intercept)
            max_degree (int): Maximum interaction degree (1 for additive, 2 for pairwise, etc.)
            threshold (float): Threshold for hinge function activation
            pairwise_terms (bool): Whether to include pairwise interaction terms
        """
        super(MARSLayer, self).__init__()
        self.n_features = n_features
        self.max_terms = max_terms
        self.max_degree = min(max_degree, 2)  # Practical limit for interpretation
        self.threshold = threshold
        self.pairwise_terms = pairwise_terms
        
        # Initialize basis functions (stored as parameters)
        self.basis_functions = nn.ModuleList([self._create_intercept_term()])
        
        # Initial linear terms
        for _ in range(n_features):
            if len(self.basis_functions) < max_terms:
                self.basis_functions.append(self._create_linear_term())
        
        # Optional pairwise terms
        if pairwise_terms and max_degree > 1 and len(self.basis_functions) < max_terms:
            for _ in range(n_features * (n_features - 1) // 2):
                if len(self.basis_functions) < max_terms:
                    self.basis_functions.append(self._create_pairwise_term())
        
        # Coefficients for each basis function
        self.coefficients = nn.Parameter(
            torch.randn(len(self.basis_functions)) * 0.01
        )
        
    def _create_intercept_term(self):
        """Create the intercept (constant) basis function"""
        return _ConstantBasisFunction()
    
    def _create_linear_term(self):
        """Create a linear hinge function basis function"""
        return _LinearHingeFunction(
            n_features=self.n_features,
            threshold=self.threshold
        )
    
    def _create_pairwise_term(self):
        """Create a pairwise interaction basis function"""
        return _PairwiseInteractionFunction(
            n_features=self.n_features,
            threshold=self.threshold
        )
    
    def forward(self, x):
        """
        Forward pass of MARS layer
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features)
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Evaluate all basis functions
        basis_outputs = [bf(x) for bf in self.basis_functions]
        basis_matrix = torch.stack(basis_outputs, dim=1)
        
        # Linear combination of basis functions
        return torch.matmul(basis_matrix, self.coefficients)
    
    def prune_terms(self, x, y, tolerance=1e-3):
        """
        Prune insignificant basis functions using a simple backward pass
        
        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor): Target values
            tolerance (float): Threshold for term removal
        """
        with torch.no_grad():
            # Get current predictions
            preds = self.forward(x)
            current_loss = nn.functional.mse_loss(preds, y)
            
            # Test removing each term
            to_keep = []
            for i, bf in enumerate(self.basis_functions):
                # Skip intercept
                if i == 0:
                    to_keep.append(True)
                    continue
                
                # Test without this term
                temp_coeff = self.coefficients.data.clone()
                temp_coeff[i] = 0.0
                temp_preds = torch.matmul(
                    torch.stack([bf(x) for bf in self.basis_functions], dim=1),
                    temp_coeff
                )
                temp_loss = nn.functional.mse_loss(temp_preds, y)
                
                # Keep if loss increases significantly
                to_keep.append((current_loss - temp_loss) > tolerance)
            
            # Update basis functions and coefficients
            self.basis_functions = nn.ModuleList(
                [bf for bf, keep in zip(self.basis_functions, to_keep) if keep]
            )
            self.coefficients = nn.Parameter(
                self.coefficients.data[torch.tensor(to_keep)]
            )
    
    def describe(self):
        """Print a description of the model terms"""
        print("MARS Model Description:")
        print(f"Number of terms: {len(self.basis_functions)}")
        print("Terms:")
        for i, (bf, coef) in enumerate(zip(self.basis_functions, self.coefficients)):
            print(f"Term {i}: Coef={coef.item():.3f} | {bf.describe()}")


class _ConstantBasisFunction(nn.Module):
    """Constant (intercept) basis function"""
    def __init__(self):
        super(_ConstantBasisFunction, self).__init__()
        
    def forward(self, x):
        return torch.ones(x.shape[0], device=x.device)
    
    def describe(self):
        return "Intercept (1.0)"


class _LinearHingeFunction(nn.Module):
    """Linear hinge function basis function"""
    def __init__(self, n_features, threshold=0.01):
        super(_LinearHingeFunction, self).__init__()
        self.n_features = n_features
        self.threshold = threshold
        
        # Randomly select a feature and knot position
        self.feature_idx = np.random.randint(0, n_features)
        self.knot = nn.Parameter(torch.randn(1) * 0.5)
        self.direction = 1 if np.random.random() > 0.5 else -1  # Randomly select hinge direction
        
    def forward(self, x):
        # Apply hinge function to selected feature
        x_feature = x[:, self.feature_idx]
        hinge = torch.max(
            torch.zeros_like(x_feature),
            self.direction * (x_feature - self.knot)
        )
        return hinge
    
    def describe(self):
        dir_str = "max(0, x" if self.direction > 0 else "max(0, -x"
        return f"Hinge on X{self.feature_idx}: {dir_str} - {self.knot.item():.3f})"


class _PairwiseInteractionFunction(nn.Module):
    """Pairwise interaction basis function (product of two hinge functions)"""
    def __init__(self, n_features, threshold=0.01):
        super(_PairwiseInteractionFunction, self).__init__()
        self.n_features = n_features
        self.threshold = threshold
        
        # Randomly select two different features
        self.feature_idxs = np.random.choice(
            n_features, 
            size=2, 
            replace=False
        )
        self.knots = nn.ParameterList([
            nn.Parameter(torch.randn(1) * 0.5),
            nn.Parameter(torch.randn(1) * 0.5)
        ])
        self.directions = [
            1 if np.random.random() > 0.5 else -1,
            1 if np.random.random() > 0.5 else -1
        ]
        
    def forward(self, x):
        # Get hinge functions for both features
        hinge1 = torch.max(
            torch.zeros_like(x[:, self.feature_idxs[0]]),
            self.directions[0] * (x[:, self.feature_idxs[0]] - self.knots[0])
        )
        hinge2 = torch.max(
            torch.zeros_like(x[:, self.feature_idxs[1]]),
            self.directions[1] * (x[:, self.feature_idxs[1]] - self.knots[1])
        )
        return hinge1 * hinge2
    
    def describe(self):
        dir1 = "max(0, x" if self.directions[0] > 0 else "max(0, -x"
        dir2 = "max(0, x" if self.directions[1] > 0 else "max(0, -x"
        return (f"Interaction X{self.feature_idxs[0]}({dir1} - {self.knots[0].item():.3f}) * "
                f"X{self.feature_idxs[1]}({dir2} - {self.knots[1].item():.3f})")



class MARSModel(nn.Module):
    def __init__(self, n_features):
        super(MARSModel, self).__init__()
        self.mars = MARSLayer(n_features, max_terms=15)
        
    def forward(self, x):
        return self.mars(x)