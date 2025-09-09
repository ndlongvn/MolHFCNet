from functools import partial
import torch
import torch.nn as nn
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import GATConv, global_add_pool, GCNConv, TransformerConv, GINEConv
from models.utils import Classification_Module
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from models.infonce import InfoNCE


class GINWrapper(torch.nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, hidden_dim=None, bias=True):
        super(GINWrapper, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.mlp = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.conv1 = GINEConv(self.mlp, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        return x

class GCNWrapper(torch.nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, hidden_dim=None, bias=True):
        super(GCNWrapper, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.edge_mlp = nn.Linear(edge_dim, input_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim, bias= bias)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_mlp(edge_attr) 
        row, col = edge_index  
        x = x + torch.zeros_like(x).scatter_add_(0, col.view(-1, 1).expand_as(edge_embedding), edge_embedding)
        # GCN layers
        x = self.conv1(x, edge_index).relu()
        return x


class TransformerWrapper(torch.nn.Module):
    def __init__(self,in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.0, num_heads: int = 1, bias=True):
        super(TransformerWrapper, self).__init__()
        self.gnn = TransformerConv(in_channels, out_channels, heads=num_heads, edge_dim=edge_dim, dropout=dropout, concat=True)

    def forward(self, x, edge_index, edge_attr):
        return self.gnn(x, edge_index, edge_attr)


class nHFC(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, edge_dim = 10, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        

        self.dwconv = gflayer(sum(self.dims), sum(self.dims), bias=True, edge_dim=edge_dim)

        self.proj_in = gflayer(dim, 2 * dim, edge_dim = edge_dim)

        self.proj_out = gflayer(dim, dim, edge_dim = edge_dim)

        self.pws = nn.ModuleList(
            [gflayer(self.dims[i], self.dims[i + 1],  edge_dim = edge_dim) for i in range(order - 1)]
        )

        self.scale = s

    def forward(self, x, edge_index, edge_attr):
        # x : N x H (N: number of nodes, H: hidden dimension)

        fused_x = self.proj_in(x, edge_index, edge_attr)
        # fused_x : N x 2H
    
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc, edge_index, edge_attr) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x, edge_index, edge_attr) * dw_list[i + 1]

        x = self.proj_out(x, edge_index, edge_attr)

        return x


class Block(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, layer_scale_init_value=1e-6, gnconv=nHFC, edge_dim = 10):
        super().__init__()

        self.norm1 = LayerNormGraph(dim, eps=1e-6)
        self.gnconv = gnconv(dim, edge_dim = edge_dim) # depthwise conv
        self.norm2 = LayerNormGraph(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, edge_index, edge_attr):
        # N, H = x.shape
        x = x + self.gamma1 * self.gnconv(self.norm1(x), edge_index, edge_attr)

        input = x
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma2 * x

        x = input + x
        return x
    
class LayerNormGraph(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNormGraph, self).__init__()
        self.ln = LayerNorm(dim, eps=eps)

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.ln(x)

class SequentialGraph(nn.Module):
    def __init__(self, *layers):
        super(SequentialGraph, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x

class Encoder(nn.Module):
    def __init__(self, node_dim=30, 
                 depths=[3, 3, 9, 3], edge_dim=10, base_dim=96, gflayer=None, 
                 layer_scale_init_value=1e-6, 
                 gnconv=nHFC, block=Block, **kwargs):
        super().__init__()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]

        self.dims = dims
        self.downsample_layers = nn.ModuleList()  # Stem and 3 intermediate downsampling layers
        stem = SequentialGraph(
            gflayer(node_dim, dims[0], edge_dim=edge_dim, bias=True),
            LayerNormGraph(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        for i in range(len(depths) - 1):
            # Downsample layers
            downsample_layer = SequentialGraph(
                LayerNormGraph(dims[i], eps=1e-6),
                gflayer(dims[i], dims[i+1], edge_dim=edge_dim, bias=True)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each with multiple blocks
        if not isinstance(gnconv, list):
            gnconv = [gnconv] * len(depths)
        else:
            assert len(gnconv) == 4

        for i in range(len(depths)):
            stage = SequentialGraph(
                *[block(dim=dims[i], edge_dim=edge_dim, layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = LayerNormGraph(dims[-1], eps=1e-6)  # Final norm layer

    def forward_features(self, x, edge_index, edge_attr):

        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x, edge_index, edge_attr)
            x = self.stages[i](x, edge_index, edge_attr)
        return x

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.forward_features(x, edge_index, edge_attr)
        return x


def encoder(gflayer, **kwargs):

    s = 1.0/3.0
    model = Encoder(depths=[2, 2, 2, 2], block=Block, # base_dim=64,
                   gflayer=gflayer,
                    gnconv=[
                        partial(nHFC, order=2, s=s, gflayer=gflayer),
                        partial(nHFC, order=3, s=s, gflayer=gflayer),
                        partial(nHFC, order=4, s=s, gflayer=gflayer),
                        partial(nHFC, order=5, s=s, gflayer=gflayer),
                    ],
                    **kwargs
                    )
    return model
        

class MolHFC(nn.Module):
    def __init__(self, node_dim, edge_dim, num_classes_tasks, gflayer, base_dim, regression=False):
        super(MolHFC, self).__init__()

        # share networks 
        if gflayer == "GAT":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=GATConv)
        elif gflayer == "GIN":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=GINWrapper)
        elif gflayer == "GCN":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=GCNWrapper)
        elif gflayer == "GTN":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=TransformerWrapper)
        else:
            raise ValueError(f"Unsupported gflayer: {gflayer}")
       
        # task networks    
        self.num_tasks = num_classes_tasks
        for t_id in range(self.num_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(num_features_xd=self.backbone.dims[-1], output_dim=1, regression=regression))    
       
    def forward(self, data):
        # share networks
        feats = [self.backbone(data)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id],data.batch)      
            outputs.append(output)

        return outputs





class MolHFCEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, gflayer, base_dim, pool='mean', num_mol_properties=6, fingerprint_dim=2048):
        super(MolHFCEncoder, self).__init__()

        if gflayer == "GAT":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=GATConv)
        elif gflayer == "GIN":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=GINWrapper)
        elif gflayer == "GCN":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=GCNWrapper)
        elif gflayer == "GTN":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=TransformerWrapper)
        else:
            raise ValueError(f"Unsupported gflayer: {gflayer}")
       

        self.info_nce = InfoNCE(self.backbone.dims[-1], self.backbone.dims[-1])

        self.property_head = nn.Sequential(
            nn.Linear(self.backbone.dims[-1], self.backbone.dims[-1]), 
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.dims[-1], num_mol_properties)
        )

        self.fingerprint_head = nn.Sequential(
            nn.Linear(self.backbone.dims[-1], self.backbone.dims[-1]), 
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.dims[-1], fingerprint_dim)
        )

        self.masked_node_head = nn.Sequential(
            nn.Linear(self.backbone.dims[-1], self.backbone.dims[-1]), 
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.dims[-1], 16)
        )

        self.masked_bond_head = nn.Sequential(
            nn.Linear(self.backbone.dims[-1]*2, self.backbone.dims[-1]), 
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.dims[-1], 4)
        )
        
        self.masked_pos_head = nn.Sequential(
            nn.Linear(self.backbone.dims[-1], self.backbone.dims[-1]), 
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.dims[-1], 3)
        )

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
    def forward(self, data, mask_task=False, inference=False):
        # share networks
        feats = self.backbone(data)

        org_feats = self.pool(feats, data.batch) 
        # print("org_feats", org_feats.shape)

        if inference:
            return feats
        if not mask_task:

            property_pred = self.property_head(org_feats)

            fingerprint = self.fingerprint_head(org_feats)

            return org_feats, property_pred, fingerprint
        
        else:
            # Masked prediction
            mask_node_indices = data.mask_idx
            # print("mask_node_indices", mask_node_indices)
            masked_node_pred = None
            masked_node_pos_pred = None
            if mask_node_indices is not None and torch.sum(mask_node_indices):
                masked_node_pred = self.masked_node_head(feats[mask_node_indices])
                masked_node_pos_pred = self.masked_pos_head(feats[mask_node_indices])

            mask_edge_indices_start = data.mask_edge_start
            # mask_edge_indices_end = data.mask_edge_end
            masked_bond_pred = None
            if mask_edge_indices_start is not None and torch.sum(mask_edge_indices_start):
                edge_features = torch.cat([feats[data.edge_index[0, mask_edge_indices_start]],
                                        feats[data.edge_index[1, mask_edge_indices_start]]], dim=1)
                masked_bond_pred = self.masked_bond_head(edge_features)
            return org_feats, masked_node_pred, masked_bond_pred, masked_node_pos_pred