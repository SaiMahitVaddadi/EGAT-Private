class CorrectAndSmooth(torch.nn.Module):
    def __init__(self, num_correction_layers=50, correction_alpha=0.9,
                 num_smoothing_layers=50, smoothing_alpha=0.8,
                 scale=1., autoscale=True):
        """
        Correct & Smooth from "Combining Label Propagation And Simple Models Out-performs GNNs"
        
        Args:
            num_correction_layers: Number of correction propagation steps
            correction_alpha: Correction teleport probability
            num_smoothing_layers: Number of smoothing propagation steps
            smoothing_alpha: Smoothing teleport probability
            scale: Residual scale factor
            autoscale: Whether to automatically determine scale
        """
        super().__init__()
        self.correction = LabelPropagation(num_correction_layers, correction_alpha)
        self.smoothing = LabelPropagation(num_smoothing_layers, smoothing_alpha)
        self.scale = scale
        self.autoscale = autoscale

    def forward(self, y_soft, y_true, train_mask, edge_index):
        """
        Args:
            y_soft: [num_nodes, num_classes] base model predictions
            y_true: [num_nodes] ground truth labels (only used for training nodes)
            train_mask: [num_nodes] BoolTensor indicating training nodes
            edge_index: [2, num_edges] graph connectivity
        Returns:
            [num_nodes, num_classes] refined predictions
        """
        num_classes = y_soft.size(1)
        
        # Convert to one-hot
        y_true_onehot = F.one_hot(y_true, num_classes).float()
        
        # === Correction Phase ===
        # Compute errors
        error = torch.zeros_like(y_soft)
        error[train_mask] = y_true_onehot - y_soft[train_mask]
        
        # Autoscale if enabled
        if self.autoscale:
            scaled_error = self.correction(error, edge_index, train_mask)
            sigma = error[train_mask].abs().sum() / train_mask.sum()
            scale = sigma / scaled_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000)] = 1.0
            error = error * scale
        
        # Propagate errors
        error = self.correction(error, edge_index, train_mask)
        y_soft = y_soft + self.scale * error
        
        # === Smoothing Phase ===
        y_soft = self.smoothing(y_soft, edge_index, train_mask)
        
        return y_soft

    @torch.no_grad()
    def predict(self, base_model, data):
        """Complete pipeline from base model to final predictions"""
        # Get base model predictions
        y_soft = base_model(data.x, data.edge_index).softmax(dim=-1)
        
        # Apply C&S
        refined = self(y_soft, data.y, data.train_mask, data.edge_index)
        
        return refined.argmax(dim=-1)
    


# Multi-stage C&S
class MultiStageCorrectAndSmooth(CorrectAndSmooth):
    def __init__(self, num_stages=3, **kwargs):
        super().__init__(**kwargs)
        self.num_stages = num_stages
    
    def forward(self, y_soft, y_true, train_mask, edge_index):
        for _ in range(self.num_stages):
            y_soft = super().forward(y_soft, y_true, train_mask, edge_index)
        return y_soft