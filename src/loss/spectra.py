import torch
import torch.nn as nn
import torch.nn.functional as F
class SpectraMAE(nn.Module):
    def __init__(self):
        super(SpectraMAE, self).__init__()

    def forward(self, prediction, target):
        # Compute the log-cosh loss
        mae_per_spectrum = torch.abs(prediction - target).mean(dim=1)
        loss = mae_per_spectrum.mean()
        return loss


class SpectraRMSE(nn.Module):
    def __init__(self):
        super(SpectraRMSE, self).__init__()

    def forward(self, prediction, target):
        # Compute the log-cosh loss
        mse_per_spectrum = torch.mean((prediction - target) ** 2, dim=1)
        rmse_per_spectrum = torch.sqrt(mse_per_spectrum)
        loss = rmse_per_spectrum.mean()
        return loss

class SpectraAUC(nn.Module):
    def __init__(self):
        super(SpectraAUC, self).__init__()

    def forward(self, prediction, target,x=torch.linspace(0, 1, steps=100)):
        integral_pred = torch.trapz(prediction, x)
        integral_tar = torch.trapz(target, x)
        loss = torch.abs(integral_pred - integral_tar)
        return loss

class SpectraWL(nn.Module):
    def __init__(self):
        super(SpectraWL, self).__init__()

    def forward(self, prediction, target,norm=True):
        if norm == True:
            prediction = prediction / prediction.sum(dim=1,keepdim=True)
            target = target / target.sum(dim=1,keepdim=True)
        prediction = torch.cumsum(prediction, dim=1)
        target = torch.cumsum(target, dim=1)
        loss = torch.sum(torch.abs(prediction - target),dim=1)
        return loss

class SpectraKLD(nn.Module):
    def __init__(self):
        super(SpectraKLD, self).__init__()

    def forward(self, prediction, target, norm=True):
        if norm == True:
            prediction = prediction / prediction.sum(dim=1,keepdim=True)
            target = target / target.sum(dim=1,keepdim=True)
        loss = F.kl_div(target.log(), prediction, reduction='batchmean')
        return loss

class SpectraSIDLoss(nn.Module):
    def __init__(self):
        super(SpectraRMSE, self).__init__()

    def forward(self, prediction, target, weight=1):
        # Compute the log-cosh loss
        mse_per_spectrum = torch.mean((prediction - target) ** 2, dim=1)
        rmse_per_spectrum = torch.sqrt(mse_per_spectrum)
        loss = rmse_per_spectrum.mean()
        return loss

class SpectraTMSELoss(nn.Module):
    def __init__(self):
        super(SpectraRMSE, self).__init__()

    def forward(self, prediction, target, weight=1):
        # Compute the log-cosh loss
        mse_per_spectrum = torch.mean((prediction - target) ** 2, dim=1)
        rmse_per_spectrum = torch.sqrt(mse_per_spectrum)
        loss = rmse_per_spectrum.mean()
        return loss

class SpectraSISLoss(nn.Module):
    def __init__(self):
        super(SpectraRMSE, self).__init__()

    def forward(self, prediction, target, weight=1):
        # Compute the log-cosh loss
        mse_per_spectrum = torch.mean((prediction - target) ** 2, dim=1)
        rmse_per_spectrum = torch.sqrt(mse_per_spectrum)
        loss = rmse_per_spectrum.mean()
        return loss
