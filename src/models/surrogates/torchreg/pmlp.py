class PMLPWithPredictor(nn.Module):
    def __init__(self, pmlp_config, predictor_type='bayesian_ridge', predictor_config=None):
        """
        PMLP model with flexible predictor head
        
        Args:
            pmlp_config: Dictionary of PMLP configuration parameters
            predictor_type: Type of predictor head
            predictor_config: Dictionary of predictor configuration parameters
        """
        super().__init__()
        self.pmlp = PMLP(**pmlp_config)
        
        # Initialize predictor head
        self.predictor_type = predictor_type
        if predictor_config is None:
            predictor_config = {}
        
        if predictor_type == 'bayesian_ridge':
            self.predictor = BayesianRidgeLayer(input_dim=pmlp_config['output_dim'], **predictor_config)
        elif predictor_type == 'ard':
            self.predictor = ARDRegressionLayer(input_dim=pmlp_config['output_dim'], **predictor_config)
        elif predictor_type == 'kernel_ridge':
            self.predictor = KernelRidge(**predictor_config)
        elif predictor_type == 'elasticnet':
            self.predictor = ElasticNet(**predictor_config)
        elif predictor_type == 'theil_sen':
            self.predictor = TheilSenRegressor(**predictor_config)
        elif predictor_type == 'huber':
            self.predictor = HuberRegressor(**predictor_config)
        elif predictor_type == 'ransac':
            self.predictor = RANSACRegressor(**predictor_config)
        elif predictor_type == 'omp':
            self.predictor = OrthogonalMatchingPursuit(**predictor_config)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    def forward(self, data, y=None, n_iter=1):
        # Extract graph features using PMLP
        x = self.pmlp(data.x, data.edge_index, data.batch)
        
        # Apply predictor head
        if isinstance(self.predictor, (BayesianRidgeLayer, ARDRegressionLayer)):
            # Bayesian models that support online updates
            return self.predictor(x, y, n_iter)
        else:
            # Standard predictors
            return self.predictor(x)
    
    def loss(self, pred, target):
        if hasattr(self.predictor, 'loss'):
            return self.predictor.loss(pred, target)
        elif self.predictor_type in ['bayesian_ridge', 'ard', 'kernel_ridge', 
                                   'elasticnet', 'theil_sen', 'huber', 'ransac', 'omp']:
            return F.mse_loss(pred, target)
        else:
            raise NotImplementedError(f"Loss not implemented for {self.predictor_type}")
        
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

class PMLPPredictor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, 
                 dropout=0.5, residual=False, alpha=0.5):
        """
        PMLP as a standalone predictor
        
        Args:
            num_layers: Number of propagational layers
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension size
            output_dim: Output dimension (1 for regression)
            dropout: Dropout rate
            residual: Whether to use residual connections
            alpha: Propagation weight (0-1)
        """
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.residual = residual
        
        # Feature transformation layers
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
        Returns:
            Predictions [num_nodes, output_dim]
        """
        # Initial transformation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[0](x)
        x = F.relu(x)
        
        # Propagational layers
        for i in range(1, self.num_layers + 1):
            # Save for residual connection
            res_connection = x if self.residual else None
            
            # Feature propagation
            x_prop = self.propagate_features(x, edge_index)
            
            # Combine propagated and transformed features
            x = self.alpha * x_prop + (1 - self.alpha) * x
            
            # Residual connection
            if res_connection is not None:
                x = x + res_connection
            
            # Transformation
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[i](x)
            if i < self.num_layers:
                x = F.relu(x)
        
        return x
    
    def propagate_features(self, x, edge_index):
        """Feature propagation using normalized adjacency"""
        row, col = edge_index
        deg = scatter_add(torch.ones_like(row), row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Symmetric normalized propagation
        return scatter_add(x[col] * norm.view(-1, 1), row, dim=0, dim_size=x.size(0))