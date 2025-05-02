import torch
import pandas as pd
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer # type: ignore
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
from selfies import encoder as selfies_encoder
from tqdm import tqdm  # to observe progress while training
import numpy as np 
from peft import LoraConfig, get_peft_model, TaskType,prepare_model_for_kbit_training
import argparse
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from gplearn.genetic import SymbolicRegressor
from pykan import KolmogorovArnoldNetwork



# Define fine-tuning model using Gradient Boosted Trees
class FineTuningModelGBT:
    def __init__(self, input_dim, output_dims, **gbt_params):
        self.models = {
            str(output_dim): GradientBoostingRegressor(**gbt_params) for output_dim in output_dims
        }

    def fit(self, X, y_dict):
        for output_dim, y in y_dict.items():
            self.models[str(output_dim)].fit(X, y)

    def predict(self, X, output_dim):
        return self.models[str(output_dim)].predict(X)


# Define fine-tuning model using Random Forest

class FineTuningModelRF:
    def __init__(self, input_dim, output_dims, **rf_params):
        self.models = {
            str(output_dim): RandomForestRegressor(**rf_params) for output_dim in output_dims
        }

    def fit(self, X, y_dict):
        for output_dim, y in y_dict.items():
            self.models[str(output_dim)].fit(X, y)

    def predict(self, X, output_dim):
        return self.models[str(output_dim)].predict(X)


# Define fine-tuning model using Multiple Linear Regression

class FineTuningModelMLR:
    def __init__(self, input_dim, output_dims):
        self.models = {
            str(output_dim): LinearRegression() for output_dim in output_dims
        }

    def fit(self, X, y_dict):
        for output_dim, y in y_dict.items():
            self.models[str(output_dim)].fit(X, y)

    def predict(self, X, output_dim):
        return self.models[str(output_dim)].predict(X)


# Define fine-tuning model using Symbolic Regression

class FineTuningModelSR:
    def __init__(self, input_dim, output_dims, **sr_params):
        self.models = {
            str(output_dim): SymbolicRegressor(**sr_params) for output_dim in output_dims
        }

    def fit(self, X, y_dict):
        for output_dim, y in y_dict.items():
            self.models[str(output_dim)].fit(X, y)

    def predict(self, X, output_dim):
        return self.models[str(output_dim)].predict(X)
    

# Define fine-tuning model using Kolmogorov-Arnold Networks (KAN) with pyKAN

class FineTuningModelKAN:
    def __init__(self, input_dim, output_dims, hidden_dim):
        self.models = {
            str(output_dim): KolmogorovArnoldNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            ) for output_dim in output_dims
        }

    def fit(self, X, y_dict, epochs=100, lr=0.001):
        self.optimizers = {
            str(output_dim): optim.Adam(self.models[str(output_dim)].parameters(), lr=lr)
            for output_dim in y_dict.keys()
        }
        self.criterions = {
            str(output_dim): nn.MSELoss() for output_dim in y_dict.keys()
        }

        for epoch in range(epochs):
            for output_dim, y in y_dict.items():
                self.models[str(output_dim)].train()
                self.optimizers[str(output_dim)].zero_grad()

                predictions = self.models[str(output_dim)](torch.tensor(X, dtype=torch.float32))
                loss = self.criterions[str(output_dim)](predictions, torch.tensor(y, dtype=torch.float32))
                loss.backward()
                self.optimizers[str(output_dim)].step()

    def predict(self, X, output_dim):
        self.models[str(output_dim)].eval()
        with torch.no_grad():
            return self.models[str(output_dim)](torch.tensor(X, dtype=torch.float32)).numpy()