import torch
import torch.nn as nn
from bayesian_torch.layers import LinearReparameterization, Conv1dReparameterization, ConvTranspose1dReparameterization, LSTMReparameterization
from ...models.base.nn.dnnstack import BaseNNCommands
from bayesian_torch.layers import LinearFlipout
from bayesian_torch.layers import Conv1dFlipout
from bayesian_torch.layers import ConvTranspose1dFlipout
from bayesian_torch.layers import LSTMFlipout

class BayesianLayers(BaseNNCommands):
    def __init__(self, cfg):
        super().__init__()
        self.params = cfg

    def create_bayesian_layer(self, in_dim, out_dim, prior_mean=0.0, prior_variance=1.0):
        return LinearReparameterization(
            in_features=in_dim,
            out_features=out_dim,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

    def create_bayesian_conv1d(self, in_channels, out_channels, kernel_size, prior_mean=0.0, prior_variance=1.0, **kwargs):
        return Conv1dReparameterization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            **kwargs
        )

    def create_bayesian_conv_transpose1d(self, in_channels, out_channels, kernel_size, prior_mean=0.0, prior_variance=1.0, **kwargs):
        return ConvTranspose1dReparameterization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            **kwargs
        )

    def create_bayesian_lstm(self, input_size, hidden_size, prior_mean=0.0, prior_variance=1.0, **kwargs):
        return LSTMReparameterization(
            input_size=input_size,
            hidden_size=hidden_size,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            **kwargs
        )
    
    def create_linear_flipout(self, in_dim, out_dim, prior_mean=0.0, prior_variance=1.0):
        return LinearFlipout(
            in_features=in_dim,
            out_features=out_dim,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

    def create_conv1d_flipout(self, in_channels, out_channels, kernel_size, prior_mean=0.0, prior_variance=1.0, **kwargs):
        return Conv1dFlipout(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            **kwargs
        )

    def create_conv_transpose1d_flipout(self, in_channels, out_channels, kernel_size, prior_mean=0.0, prior_variance=1.0, **kwargs):
        return ConvTranspose1dFlipout(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            **kwargs
        )

    def create_lstm_flipout(self, input_size, hidden_size, prior_mean=0.0, prior_variance=1.0, **kwargs):
        return LSTMFlipout(
            input_size=input_size,
            hidden_size=hidden_size,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            **kwargs
        )
    
    def select_bayesian_layer(self, in_dim, out_dim, **kwargs):
        layer_type = self.params.bayesiantorch_layer.lower()
        if layer_type == "linear_reparameterization":
            return self.create_bayesian_layer(in_dim, out_dim, **kwargs)
        elif layer_type == "linear_flipout":
            return self.create_linear_flipout(in_dim, out_dim, **kwargs)
        elif layer_type == "conv1d_reparameterization":
            return self.create_bayesian_conv1d(in_dim, out_dim, **kwargs)
        elif layer_type == "conv1d_flipout":
            return self.create_conv1d_flipout(in_dim, out_dim, **kwargs)
        elif layer_type == "convtranspose1d_reparameterization":
            return self.create_bayesian_conv_transpose1d(in_dim, out_dim, **kwargs)
        elif layer_type == "convtranspose1d_flipout":
            return self.create_conv_transpose1d_flipout(in_dim, out_dim, **kwargs)
        elif layer_type == "lstm_reparameterization":
            return self.create_bayesian_lstm(in_dim, out_dim, **kwargs)
        elif layer_type == "lstm_flipout":
            return self.create_lstm_flipout(in_dim, out_dim, **kwargs)
        else:
            raise ValueError(f"Unsupported bayesian layer type: {self.params.bayesiantorch_layer}")


class DeepBNN(BayesianLayers):
    def __init__(self, cfg):
        super().__init__()
        self.params = cfg
        self.create_bayesian_mlp(activation=self.params.activation, indim=self.params.indim, outdim=self.params.outdim, dropout=self.params.dropout, addons=self.params.addons)

    


    
    

    def create_bayesian_mlp(self, activation='GELU', indim=None, outdim=None, dropout=None, addons=0):
        indim, _, outdim = self._defaultdims(indim=indim, outdim=outdim)
        hidden_dim = self.hiddenlayerpropnet()
        output = self.propnetoutput()
        addon = self.addons(addons)

        self.layers = nn.ModuleList()
        input_dim = hidden_dim + addon

        for _ in range(self.params.num_layers - 1):
            layer = self.create_bayesian_layer(input_dim, self.params.propnet_hdim)
            self.layers.append(layer)
            if activation is not None:
                activation_fn = self.getactivation(activation)
                self.layers.append(activation_fn)
            if dropout is not None:
                self.layers.append(nn.Dropout(p=dropout))
            input_dim = self.params.propnet_hdim

        final_layer = self.create_bayesian_layer(input_dim, output)
        self.layers.append(final_layer)
        if activation is not None:
            activation_fn = self.getactivation(activation)
            self.layers.append(activation_fn)
        if dropout is not None:
            self.layers.append(nn.Dropout(p=dropout))

    

    def forward(self, x, addons=None, activation='ReLU'):
        if addons is not None:
            x = torch.cat((x, addons), dim=1)

        kl_divergence = 0.0
        activation_fn = None

        if isinstance(activation, str):
            activation_fn = getattr(nn, activation)()
        elif isinstance(activation, list):
            activation_fn = [getattr(nn, act)() for act in activation]

        for i, layer in enumerate(self.layers[:-1]):
            x, kl = layer(x)
            if isinstance(activation_fn, list):
                x = activation_fn[i](x) if i < len(activation_fn) else nn.ReLU()(x)
            else:
                x = activation_fn(x)
            kl_divergence += kl

        x, kl = self.layers[-1](x)
        kl_divergence += kl

        return x, kl_divergence