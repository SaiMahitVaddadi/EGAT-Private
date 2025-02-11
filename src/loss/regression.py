# custom_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLoss(nn.Module):
    def __init__(self, weight):
        super(WeightedLoss, self).__init__()
        self.weight = weight

    def forward(self, prediction, target):
        loss = (prediction - target) ** self.weight
        return loss.mean()

class PowerLoss(nn.Module):
    def __init__(self, weight,absolute = True):
        super(WeightedLoss, self).__init__()
        self.weight = weight
        self.absolute = absolute

    def forward(self, prediction, target):
        loss = self.weight ** prediction - self.weight ** target
        if self.absolute: loss = torch.abs(loss)
        return loss.mean()

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
    def forward(self, prediction, target):
        absolute_percentage_error = torch.abs((target - prediction) / target)
        loss = torch.mean(absolute_percentage_error) * 100.0
        return loss

class MSPELoss(nn.Module):
    def __init__(self):
        super(MSPELoss, self).__init__()

    def forward(self, prediction, target):
        squared_percentage_error = ((target - prediction) / target) ** 2
        loss = torch.mean(squared_percentage_error) * 100.0
        return loss

class RAELoss(nn.Module):
    def __init__(self):
        super(RAELoss, self).__init__()

    def forward(self, prediction, target):
        absolute_errors = torch.abs(target - prediction)
        absolute_diff_mean = torch.abs(target - torch.mean(target))
        rae = torch.sum(absolute_errors) / torch.sum(absolute_diff_mean)
        return rae

class RSELoss(nn.Module):
    def __init__(self):
        super(RSELoss, self).__init__()

    def forward(self, prediction, target):
        squared_errors = (target - prediction) ** 2
        squared_diff_mean = (target - torch.mean(target)) ** 2
        rse = torch.sum(squared_errors) / torch.sum(squared_diff_mean)
        return rse


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, prediction, target):
        # Compute the log-cosh loss
        loss = torch.log(torch.cosh(prediction - target))
        return torch.mean(loss)

class MSEConstrainedLoss(nn.Module):
    def __init__(self):
        super(MSEConstrainedLoss, self).__init__()

    def forward(self, prediction, target,weight=5):
        # Compute the log-cosh loss
        loss = torch.mean((prediction - target)**2) + weight * torch.sum(torch.clamp(-prediction, min=0))
        return loss

class MAEConstrainedLoss(nn.Module):
    def __init__(self):
        super(MAEConstrainedLoss, self).__init__()

    def forward(self, prediction, target,weight=5):
        # Compute the log-cosh loss
        loss = torch.mean(torch.abs(prediction - target)) + weight * torch.sum(torch.clamp(-prediction, min=0))
        return loss

class MAESoftPlusLoss(nn.Module):
    def __init__(self):
        super(MAESoftPlusLoss, self).__init__()

    def forward(self, prediction, target, weight=5):
        # Compute the log-cosh loss
        loss = torch.mean(torch.abs(prediction - target)) + weight * torch.sum(torch.log(1 + torch.exp(-prediction)))
        return loss

class MSESoftPlusLoss(nn.Module):
    def __init__(self):
        super(MSESoftPlusLoss, self).__init__()

    def forward(self, prediction, target, weight=1):
        # Compute the log-cosh loss
        loss = torch.mean((prediction - target)**2) + weight * torch.sum(torch.log(1 + torch.exp(-prediction)))
        return loss

class MAEActLSLoss(nn.Module):
    def __init__(self):
        super(MAEActLSLoss, self).__init__()

    def forward(self, prediction, target, weight=5):
        # Compute the log-cosh loss
        m = nn.LogSigmoid()
        loss = torch.mean(torch.abs(prediction - target)) + weight * -m(-prediction)
        return loss

class MSEActLSLoss(nn.Module):
    def __init__(self):
        super(MSEActLSLoss, self).__init__()

    def forward(self, prediction, target, weight=1):
        # Compute the log-cosh loss
        m = nn.LogSigmoid()
        loss = torch.mean((prediction - target)**2) + weight * -m(-prediction)
        return loss

class MAEActSiLoss(nn.Module):
    def __init__(self):
        super(MAEActLSLoss, self).__init__()

    def forward(self, prediction, target, weight=5):
        # Compute the log-cosh loss
        m = nn.SiLU()
        loss = torch.mean(torch.abs(prediction - target)) + weight * -m(-prediction)
        return loss

class MSEActSiLoss(nn.Module):
    def __init__(self):
        super(MSEActLSLoss, self).__init__()

    def forward(self, prediction, target, weight=1):
        # Compute the log-cosh loss
        m = nn.SiLU()
        loss = torch.mean((prediction - target)**2) + weight * -m(-prediction)
        return loss

class MAEActSiLoss(nn.Module):
    def __init__(self):
        super(MAEActLSLoss, self).__init__()

    def forward(self, prediction, target, weight=5):
        # Compute the log-cosh loss
        m = nn.SiLU()
        loss = torch.mean(torch.abs(prediction - target)) + weight * -m(-prediction)
        return loss

class MSEActSiLoss(nn.Module):
    def __init__(self):
        super(MSEActLSLoss, self).__init__()

    def forward(self, prediction, target, weight=1):
        # Compute the log-cosh loss
        m = nn.SiLU()
        loss = torch.mean((prediction - target)**2) + weight * -m(-prediction)
        return loss

class MAEActGELULoss(nn.Module):
    def __init__(self):
        super(MAEActLSLoss, self).__init__()

    def forward(self, prediction, target, weight=5):
        # Compute the log-cosh loss
        m = nn.GELU()
        loss = torch.mean(torch.abs(prediction - target)) + weight * -m(-prediction)
        return loss

class MSEActGELULoss(nn.Module):
    def __init__(self):
        super(MSEActLSLoss, self).__init__()

    def forward(self, prediction, target, weight=1):
        # Compute the log-cosh loss
        m = nn.GELU()
        loss = torch.mean((prediction - target)**2) + weight * -m(-prediction)
        return loss

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
