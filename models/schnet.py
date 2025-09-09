import os
import os.path as osp
from math import pi as PI

import ase
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.utils import scatter
from torch_geometric.nn import radius_graph
from typing import Optional


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels,
                           num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x
    


class SchNetLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = None, # = 128,
        num_filters: int = 128,
        num_interactions: int = 1,
        num_gaussians: int = 51,
        cutoff: float = 10.0,
        readout: str = 'mean',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            in_channels (int): Number of input features per node.
            out_channels (int): Number of output features per node.
            hidden_channels (int): Number of hidden features in the interaction blocks.
            num_filters (int): Number of filters in the interaction blocks.
            num_interactions (int): Number of interaction blocks.
            num_gaussians (int): Number of Gaussian functions for distance expansion.
            cutoff (float): Cutoff distance for neighbor search.
            readout (str): Readout method ("add", "sum", "mean").
            dipole (bool): Whether to compute dipole moments.
            mean (float, optional): Mean value for normalization.
            std (float, optional): Standard deviation for normalization.
            atomref (torch.Tensor, optional): Atomic reference values.
        """
        super(SchNetLayer, self).__init__()

        assert readout in ['add', 'sum', 'mean']
        
        if hidden_channels is None:
            hidden_channels = in_channels
            
        # print('in_channels',in_channels)
        # print('out_channels',out_channels)
        # print('hidden_channels',hidden_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = 'add' if dipole else readout
        self.dipole = dipole
        self.mean = mean
        self.std = std

        # Atomic mass buffer
        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        # Distance expansion
        # self.embedding = Embedding(120, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

        # Linear transformations
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, out_channels)

        # Atom reference
        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        for interaction in self.interactions:
            interaction.reset_parameters()
        glorot(self.lin1.weight)
        zeros(self.lin1.bias)
        glorot(self.lin2.weight)
        zeros(self.lin2.bias)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert z.dim() == 2 and z.size(1) == self.in_channels, f'Expected z to have shape [*, {self.in_channels}], got {z.size()}'
        # assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device) if batch is None else batch

        # Initialize node features
        # h = self.embedding(z)
        h = z

        # Compute edges and edge attributes
    
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        # Apply interaction blocks
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Final transformations
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        # Dipole moment computation
        if self.dipole:
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        # Normalization
        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        # Atom reference
        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)
        

        # # Readout
        # out = scatter(h, batch, dim=0, reduce=self.readout)

        # # Dipole norm
        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)
  
        return h

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels,
                           num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
