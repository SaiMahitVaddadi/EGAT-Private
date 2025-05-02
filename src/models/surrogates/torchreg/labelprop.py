import torch
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor

class LabelPropagation(torch.nn.Module):
    def __init__(self, num_layers=50, alpha=0.9, normalize=True):
        """
        Label Propagation from "Learning from Labeled and Unlabeled Data with Label Propagation"
        
        Args:
            num_layers: Number of propagation steps
            alpha: Teleport probability (1 - alpha = restart probability)
            normalize: Whether to use symmetric normalized Laplacian
        """
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, y, edge_index, mask=None):
        """
        Args:
            y: [num_nodes, num_classes] FloatTensor of labels (use one-hot for known labels)
            edge_index: [2, num_edges] LongTensor of graph edges
            mask: [num_nodes] BoolTensor indicating which nodes to keep fixed
        Returns:
            [num_nodes, num_classes] predictions
        """
        if mask is None:
            mask = torch.zeros(y.size(0), dtype=torch.bool)
        
        # Convert to sparse tensor for efficient propagation
        edge_index, _ = get_laplacian(edge_index, normalization='sym' if self.normalize else None)
        adj = SparseTensor.from_edge_index(edge_index).to(y.device)
        
        result = y.clone()
        for _ in range(self.num_layers):
            # Propagate labels
            result = adj @ result
            
            # Keep original labels for masked nodes
            result[mask] = y[mask]
            
            # Apply teleportation probability
            result = self.alpha * result + (1 - self.alpha) * y
        
        return result

    @torch.no_grad()
    def predict(self, y_train, train_mask, edge_index, num_classes):
        """Helper method for semi-supervised learning"""
        y = torch.zeros((edge_index.max()+1, num_classes), device=y_train.device)
        y[train_mask] = F.one_hot(y_train, num_classes).float()
        return self(y, edge_index, train_mask)
    

# Custom propagation matrices
def custom_propagation(edge_index):
    # Create your own propagation matrix (e.g., PPR, heat kernel)
    pass

# Modify the LabelPropagation class to use custom propagation
class CustomLabelPropagation(LabelPropagation):
    def forward(self, y, edge_index, mask=None):
        adj = custom_propagation(edge_index)
        # ... rest remains same
