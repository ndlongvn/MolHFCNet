import math
from typing import Any
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class Classification_Module(nn.Module):
    def __init__(self, num_features_xd=256, output_dim=1, dropout = 0.1, regression=False, pool='max'):
        super(Classification_Module, self).__init__()
        self.fc_g1   = torch.nn.Linear(num_features_xd, num_features_xd)
        self.fc_g2   = torch.nn.Linear(num_features_xd, num_features_xd//2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out     = nn.Linear(num_features_xd//2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.regression = regression
            
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        

    def forward(self, x, batch):
        # Maxpool
        x   = self.pool(x, batch)
        #MLP
        x   = self.relu(self.fc_g1(x))
        x   = self.dropout(x)
        x   = self.fc_g2(x)
        x   = self.dropout(x)
        out = self.out(x).view(-1)
        if self.regression:
            return out
        out = self.sigmoid(out)
        return out



def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)


def kaiming_uniform(value: Any, fan: int, a: float):
    if isinstance(value, Tensor):
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            kaiming_uniform(v, fan, a)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            kaiming_uniform(v, fan, a)


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.)


def ones(tensor: Any):
    constant(tensor, 1.)


def normal(value: Any, mean: float, std: float):
    if isinstance(value, Tensor):
        value.data.normal_(mean, std)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            normal(v, mean, std)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            normal(v, mean, std)


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)
