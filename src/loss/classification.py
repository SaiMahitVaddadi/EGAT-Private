import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F


class ROCAUCLoss(nn.Module):
    def __init__(self):
        super(ROCAUCLoss, self).__init__() 

    def forward(self, y_pred, y_true):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        return 1 - roc_auc_score(y_true, y_pred)