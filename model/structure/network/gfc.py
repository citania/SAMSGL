import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from ..graphconv import GraphConv
from .layers import GenericBasis, Align_series



class GFC_block(nn.Module):
    r"""An implementation of the Multi-Component Spatial-Temporal Graph
    Convolution block `_

    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
    """

    def __init__(self, in_channel, out_channel: int, max_view: int, nb_chev_filter: int, conv_layers,
                 seq_len, horizon, layers, layer_size, theta_size, align: nn.Module):
        #theta_size = input_size + output_size
        super(GFC_block, self).__init__()

        self.horizon = horizon
        self.out_dim = out_channel

        self.align = align
        self.conv_layer = nn.ModuleList([GraphConv(
            input_dim=nb_chev_filter,
            output_dim=nb_chev_filter,
            max_view=max_view,
        align = align)]+
        [GraphConv(
            input_dim=nb_chev_filter,
            output_dim=nb_chev_filter,
            max_view=max_view,
        align = align) for _ in range(conv_layers - 1)])

        self.layers = nn.ModuleList([nn.Linear(in_features=nb_chev_filter*seq_len,out_features=layer_size)]+
                                    [nn.Linear(in_features=layer_size,out_features=layer_size)
                                    for _ in range(layers - 1)])
        # self._layer_norm = nn.LayerNorm(nb_time_filter)
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = GenericBasis(backcast_size=seq_len*nb_chev_filter,
                                           forecast_size=horizon*out_channel)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, kernel) -> torch.FloatTensor:

        batch_size, num_of_timesteps, num_of_vertices, in_channels = X.shape

        # X_t, delay,weight, over_x = self.align(X,X,X)
        # print(X.size(), delay.size())
        block_input = X
        for block in self.conv_layer:
            block_input = F.relu(block(block_input, kernel))
        # block_input = self.align.backward(block_input, delay)
        out_channel = block_input.shape[-1]
        block_input = block_input.permute(0,2,1,3).reshape([batch_size * num_of_vertices, num_of_timesteps * out_channel])

        for block in self.layers:
            block_input = F.relu(block(block_input))
        basis_parameters = self.basis_parameters(block_input)
        b, f = self.basis_function(basis_parameters)
        b = b.reshape([batch_size, num_of_vertices, num_of_timesteps, out_channel])
        f = f.reshape([batch_size, num_of_vertices, self.horizon, self.out_dim])
        return b.permute(0, 2, 1, 3), f.permute(0, 2, 1, 3)


class SAMSGL_NET(nn.Module):
    def __init__(self, nb_block: int, input_dim: int, output_dim: int, max_view: int, nb_chev_filter: int,
                 seq_len: int, horizon: int, dynamic_graph, embed_dim=None,
                 **model_kwargs):
        super(SAMSGL_NET, self).__init__()

        n_head = model_kwargs['n_head']
        self.horizon = horizon
        self.output_dim = output_dim
        self.dynamic_graph = dynamic_graph

        if embed_dim is not None:
            self.input_dim = input_dim + embed_dim * 2
        ####
        self.embed_fit = nn.Linear(in_features=self.input_dim, out_features=nb_chev_filter)
        # self.align = Align_series(n_heads=8, d_model=nb_chev_filter)
        self._blocklist = nn.ModuleList(
            [GFC_block(self.input_dim, self.output_dim, max_view, nb_chev_filter, model_kwargs["conv_layer"],
                       seq_len, horizon, model_kwargs["time_layer"], model_kwargs["layer_size"],
                       theta_size=seq_len*nb_chev_filter+horizon*output_dim, align=Align_series(n_heads=8, d_model=nb_chev_filter,
                            time_layer=weight_norm(nn.Conv1d(nb_chev_filter//8*seq_len, nb_chev_filter//8*seq_len, 3,
                                           stride=1, padding=1,))))])

        self._blocklist.extend(
            [GFC_block(self.input_dim, self.output_dim, max_view, nb_chev_filter, model_kwargs["conv_layer"],
                       seq_len, horizon, model_kwargs["time_layer"], model_kwargs["layer_size"],
                       theta_size=seq_len*nb_chev_filter+horizon*output_dim, align=Align_series(n_heads=n_head, d_model=nb_chev_filter,
                                                                                                 time_layer=weight_norm(
                                                                                                     nn.Conv1d(
                                                                                                         nb_chev_filter // n_head * seq_len,
                                                                                                         nb_chev_filter // n_head * seq_len,
                                                                                                         3,
                                                                                                         stride=1,
                                                                                                         padding=1, )
                                                                                                     ) ))
             for _ in range(nb_block - 1)])
        ####
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, kernel) -> torch.FloatTensor:
        # kernel g1
        T_in, B, N_nodes, F_in = X.shape
        F_out = self.output_dim
        T_out = self.horizon
        residuals = X.permute(1, 0, 2, 3)#B, T_in, N_nodes, F_in
        residuals = self.embed_fit(residuals)
        forecast = 0

        for block in self._blocklist:
            X_back, X_head = block(residuals, kernel)
            residuals = (residuals - X_back)
            forcast = forecast + X_head



        return forcast.permute(1, 0, 2, 3)