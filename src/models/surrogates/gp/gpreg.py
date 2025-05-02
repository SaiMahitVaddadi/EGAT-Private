import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
import torch

# Define Gaussian Process Model
class GaussianProcessModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GaussianProcessModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define Non-Gaussian Gaussian Process Model
class NonGaussianProcessModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(NonGaussianProcessModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Apply a non-Gaussian transformation (e.g., softplus)
        transformed_mean = torch.nn.functional.softplus(mean_x)
        return gpytorch.distributions.MultivariateNormal(transformed_mean, covar_x)