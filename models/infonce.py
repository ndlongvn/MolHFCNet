import torch
import torch.nn.functional as F
from torch import nn
import torch

class InfoNCE(nn.Module):
    
    def __init__(self, bert_output_size, graph_ouput_size, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.orig_d_l = bert_output_size
        self.orig_d_av = graph_ouput_size
        self.d_l, self.d_av = 50, 50 ### 50, 50, 30, 30
        # self.d_l, self.d_av = 30, 30 ### 50, 50, 30, 30
        self.embed_dropout = 0.1
        self.training = True

        self.info_proj_query = nn.Sequential(nn.Linear(self.orig_d_l, self.orig_d_l), nn.ReLU(), nn.Linear(self.orig_d_l, self.d_l))
        self.info_proj_positive = nn.Sequential(nn.Linear(self.orig_d_av, self.orig_d_av), nn.ReLU(), nn.Linear(self.orig_d_av, self.d_av))

    def forward(self, query, positive_key, negative_keys=None):
        x_l_ = F.dropout(query, p=self.embed_dropout, training=self.training)
        x_av_ = positive_key

        # Project the textual/visual/audio features
        proj_x_l = x_l_ if self.orig_d_l == self.d_l else self.info_proj_query(x_l_)
        proj_x_av = x_av_ if self.orig_d_av == self.d_av else self.info_proj_positive(x_av_)

        ###消除序列长度的影响,做成二维的方式
        # proj_query = torch.mean(proj_x_l, dim=1)
        # proj_positive = torch.mean(proj_x_av, dim=1)

        return info_nce(proj_x_l, proj_x_av, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    # query dim != positive_key dim
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    return (contrastive_loss(query, positive_key, temperature)+ contrastive_loss(positive_key, query, temperature))/2


def contrastive_loss(z1, z2, temperature=0.5):

    # Similarity matrix between z1 and z2
    sim_matrix = torch.matmul(z1, z2.T) / temperature

    # Positive pairs are on the diagonal
    positive_sim = torch.exp(torch.diag(sim_matrix))
    
    # Sum over rows (denominator)
    all_sim = torch.exp(sim_matrix).sum(dim=1)

    # Loss: -log(pos / total)
    loss = -torch.log(positive_sim / all_sim)
    return loss.mean()


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


