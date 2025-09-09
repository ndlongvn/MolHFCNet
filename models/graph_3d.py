from functools import partial
import torch
import torch.nn as nn
from torch_geometric.nn.norm import LayerNorm
# from models.molecule_gnn_model import GINConv, GATConv, GCNConv, TransformerConv, num_atom_type, num_chirality_tag, num_bond_type, num_bond_direction
from models.schnet import SchNetLayer
from models.sparse_conv import SpatialGraphConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from models.utils import Classification_Module
from models.infonce import InfoNCE

   
class nHFC(nn.Module):
    def __init__(self, dim, order=5, edge_dim=10, gflayer=SchNetLayer, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
       
        if gflayer is None:
            raise ValueError("Graph convolution layer (gflayer) must be provided.")
       
        # Graph convolution for 2D graph
        self.dwconv = gflayer(sum(self.dims), sum(self.dims))
     
        # Projections for input/output
        self.proj_in = gflayer(dim, 2 * dim)
        self.proj_out = gflayer(dim, dim)
       
        # Pairwise convolutions
        self.pws = nn.ModuleList(
            [gflayer(self.dims[i], self.dims[i + 1]) for i in range(order - 1)]
        )
       
        self.scale = s
        # print('[GNConv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale) 

    def forward(self, x, x_3d, pos, batch=None, edge_index=None):
       
        # 2D graph feature transformation
        fused_x = self.proj_in(x_3d, pos, batch, edge_index)  # N x 2H
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc, pos, batch, edge_index) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
   
        x = pwa*dw_list[0]
       
        for i in range(self.order - 1):
            x_1 = self.pws[i](x, pos, batch, edge_index) # M1 * N1
   
            x = x_1 * dw_list[i + 1]

        x = self.proj_out(x, pos, batch, edge_index)
       
        return x
   


class Block(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, layer_scale_init_value=1e-6, gnconv=nHFC, edge_dim = 10):
        super().__init__()

        self.norm1 = LayerNormGraph(dim, eps=1e-6)
        self.gnconv = gnconv(dim) # depthwise conv
        self.norm2 = LayerNormGraph(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x, x_3d, pos, batch=None, edge_index=None):
        # N, H = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x), x_3d, pos, batch, edge_index))

        input = x
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x

        x = input + self.drop_path(x)
        return x
   
class LayerNormGraph(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNormGraph, self).__init__()
        self.ln = LayerNorm(dim, eps=eps)

    def forward(self, x, x_3d=None, pos=None, batch=None, edge_index=None):
        return self.ln(x)

class SequentialGraph(nn.Module):
    def __init__(self, *layers):
        super(SequentialGraph, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, pos=None, batch=None, edge_index=None):
        for layer in self.layers:
            x = layer(x, pos, batch, edge_index)
        return x
   
class SequentialGraphAdd(nn.Module):
    def __init__(self, *layers):
        super(SequentialGraphAdd, self).__init__()
        self.layers = nn.ModuleList(layers)
   
    def forward(self, x, x_3d=None, pos=None, batch=None, edge_index=None):
        for layer in self.layers:
            x = layer(x, x_3d, pos, batch, edge_index)
        return x

class Encoder(nn.Module):
    def __init__(self, node_dim=30,
                 depths=[3, 3, 9, 3], edge_dim=10, base_dim=96, gflayer=None, geolayer=None,
                 layer_scale_init_value=1e-6,
                 gnconv=nHFC, block=Block, **kwargs):
        super().__init__()
       
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]

        self.dims = dims
        self.downsample_layers = nn.ModuleList()  # Stem and 3 intermediate downsampling layers
        stem = SequentialGraph(
            gflayer(node_dim, dims[0]),
            LayerNormGraph(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        for i in range(len(depths) - 1):
            # Downsample layers
            downsample_layer = SequentialGraph(
                LayerNormGraph(dims[i], eps=1e-6),
                gflayer(dims[i], dims[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each with multiple blocks
        if not isinstance(gnconv, list):
            gnconv = [gnconv] * len(depths)
        else:
            assert len(gnconv) == len(depths)

        for i in range(len(depths)):
            stage = SequentialGraphAdd(
                *[block(dim=dims[i], edge_dim=edge_dim, layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = LayerNormGraph(dims[-1], eps=1e-6)  # Final norm layer

    def forward_features(self, x, pos, edge_index, batch):
       
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x, pos, batch, edge_index)
            x = self.stages[i](x, x, pos, batch, edge_index)
        return x

    def forward(self, data):
        x, edge_index, pos, batch= data.x, data.edge_index, data.positions, data.batch
        x = self.forward_features(x, pos, edge_index, batch)
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
   



class MolHFCEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, gflayer, base_dim, pool='mean', num_mol_properties=6, fingerprint_dim=2048):
        super(MolHFCEncoder, self).__init__()

        
        if gflayer == "CFC":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=SchNetLayer)

        elif gflayer == "SGCN":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=SpatialGraphConv)
       

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

        if inference:
            return org_feats
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




class MolHFC(nn.Module):
    def __init__(self, node_dim, edge_dim, num_classes_tasks, gflayer, base_dim, regression=False, pool='mean'):
        super(MolHFC, self).__init__()

        # share networks 
        if gflayer == "CFC":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=SchNetLayer)

        elif gflayer == "SGCN":
            self.backbone = encoder(edge_dim = edge_dim, node_dim = node_dim, base_dim = base_dim, gflayer=SpatialGraphConv)
     
       
        # task networks
           
        self.num_tasks = num_classes_tasks
        for t_id in range(self.num_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(num_features_xd=base_dim*8, output_dim=1, regression=regression))    
       
    def forward(self, data):
        # share networks
        feats = [self.backbone(data)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id],data.batch)      
            outputs.append(output)

        return outputs