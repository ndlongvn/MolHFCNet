import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops
import torch
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import graclus, max_pool, global_mean_pool
from torch_geometric.data import Batch, Data
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))





class SpatialGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, coors=3, hidden_size=None, dropout=0):
        """
        coors - dimension of positional descriptors (e.g. 2 for 2D images)
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        hidden_size - number of the inner convolutions
        dropout - dropout rate after the layer
        """
        super(SpatialGraphConv, self).__init__(aggr='add')
        self.dropout = dropout
        if hidden_size is None:
            hidden_size = 16
        self.lin_in = torch.nn.Linear(coors, hidden_size * in_channels)
        self.lin_out = torch.nn.Linear(hidden_size * in_channels, out_channels)
        self.in_channels = in_channels

    def forward(self, x, pos, batch, edge_index):
        """
        x - feature matrix of the whole graph [num_nodes, label_dim]
        pos - node position matrix [num_nodes, coors]
        edge_index - graph connectivity [2, num_edges]
        """
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # num_edges = num_edges + num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        out =  self.propagate(edge_index=edge_index, x=x, pos=pos, aggr='add')  # [N, out_channels, label_dim]
        
        return out

    def message(self, pos_i, pos_j, x_j):
        """
        pos_i [num_edges, coors]
        pos_j [num_edges, coors]
        x_j [num_edges, label_dim]
        """

        relative_pos = pos_j - pos_i  # [n_edges, hidden_size * in_channels]
        spatial_scaling = F.relu(self.lin_in(relative_pos))  # [n_edges, hidden_size * in_channels]

        n_edges = spatial_scaling.size(0)
        # [n_edges, in_channels, ...] * [n_edges, in_channels, 1]
        result = spatial_scaling.reshape(n_edges, self.in_channels, -1) * x_j.unsqueeze(-1)
        return result.view(n_edges, -1)

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        aggr_out = self.lin_out(aggr_out)  # [num_nodes, label_dim, out_features]
        aggr_out = F.relu(aggr_out)
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)

        return aggr_out
    
    
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import graclus, max_pool, global_mean_pool



class SGCN(torch.nn.Module):
    def __init__(self, dim_coor, out_dim, input_features,
                 layers_num, model_dim, out_channels_1, dropout,
                 use_cluster_pooling):
        super(SGCN, self).__init__()
        self.layers_num = layers_num
        self.use_cluster_pooling = use_cluster_pooling

        self.conv_layers = [SpatialGraphConv(coors=dim_coor,
                                             in_channels=input_features,
                                             out_channels=model_dim,
                                             hidden_size=out_channels_1,
                                             dropout=dropout)] + \
                           [SpatialGraphConv(coors=dim_coor,
                                             in_channels=model_dim,
                                             out_channels=model_dim,
                                             hidden_size=out_channels_1,
                                             dropout=dropout) for _ in range(layers_num - 1)]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        self.fc1 = torch.nn.Linear(model_dim, out_dim)

    def forward(self, data):
        for i in range(self.layers_num):
            data.x = self.conv_layers[i](data.x, data.pos, data.edge_index)

            if self.use_cluster_pooling:
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = global_mean_pool(data.x, data.batch)
        x = self.fc1(data.x)

        return F.log_softmax(x, dim=1)