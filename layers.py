from inspect import stack
import torch
import torch.nn as nn
import math

from torch.nn import Linear, LSTM, init, LayerNorm, BatchNorm1d
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNNConv(nn.Module):
    def __init__(self, task, conv_name, in_channels, out_channels, norm,
                 n_heads=[1, 1], iscat=[False, False], dropout_att=0.):
        super(GNNConv, self).__init__()
        self.task = task
        self.name = conv_name

        if self.name == 'gcn_conv':
            self.conv  = GCNConv(in_channels, out_channels)
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.linear = nn.Linear(in_channels, out_channels)

        elif self.name == 'sage_conv':
            self.conv  = SAGEConv(in_channels, out_channels)
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.linear = nn.Linear(in_channels, out_channels)

        elif self.name == 'gat_conv':
            if iscat[0]: # if previous gatconv's cat is True
                in_channels = in_channels * n_heads[0]
            self.conv  = GATConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 heads=n_heads[1],
                                 concat=iscat[1],
                                 dropout=dropout_att)
            if iscat[1]: # if this gatconv's cat is True
                out_channels = out_channels * n_heads[1]

            if self.task == 'inductive':  # if transductive, we dont use linear
                self.linear = nn.Linear(in_channels, out_channels)
        
        if norm == 'LayerNorm':
            self.norm = LayerNorm(out_channels)
        elif norm == 'BatchNorm1d':
            self.norm = BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()


    def forward(self, x, edge_index):
        if self.task == 'transductive':
            x = self.conv(x, edge_index)
        elif self.task == 'inductive':
            x = self.conv(x, edge_index) + self.linear(x)

        return self.norm(x)


def query_transform(query, weights, summary_mode):
    n_hid = query.shape[2]
    trans_matrices = []

    if summary_mode == 'vanilla':
        buff = torch.eye(n_hid).to(query.device.type) # Identity matrix
        for weight in reversed(weights[1:]): # [W^L, W^(L-1), ..., W^2]
            trans_matrices.append(buff)
            buff = torch.mm(buff, weight)
        trans_matrices.append(buff)
        trans_matrices = list(reversed(trans_matrices))
        trans_matrices = torch.stack(trans_matrices, dim=0)
        
    else: # if summary_mode == roll
        for weight in weights[1:]: # [W^2, W^3, ..., W^L]
            trans_matrices.append(weight)
        trans_matrices = torch.stack(trans_matrices, dim=0)

    query_transformed = torch.bmm(query.permute(1,0,2), trans_matrices)

    return query_transformed.permute(1,0,2)


def weight_transform(weight_l, weight_r, weights, n_nodes, summary_mode):
    n_hid = weight_l.shape[0]
    n_layer = len(weights)

    trans_matrices_l, trans_matrices_r = [], []
    buff = torch.eye(n_hid).to(weight_l.device.type)
    for weight in weights:
        buff = torch.mm(buff, weight)
        trans_matrices_l.append(torch.mm(buff, weight_l))
        trans_matrices_r.append(torch.mm(buff, weight_r))
        
    # trans_matrices_l is [W^1w_l, W^1W^2w_l, W^1W^2W^3w_l, ...]
    if summary_mode == 'vanilla':
        trans_matrices_l = [trans_matrices_l[-1] for _ in range(n_layer)]
        trans_matrices_r = [trans_matrices_r[-1] for _ in range(n_layer)]
    else: # if summary_mode == roll
        trans_matrices_l = trans_matrices_l[1:]
        trans_matrices_r = trans_matrices_r[1:]
    weight_l_transformed = torch.stack(trans_matrices_l, dim=0).expand(-1, -1, n_nodes).permute(2, 0, 1)
    weight_r_transformed = torch.stack(trans_matrices_r, dim=0).expand(-1, -1, n_nodes).permute(2, 0, 1)

    return weight_l_transformed, weight_r_transformed


# if cfg.skip_connection is summarize
class SummarizeSkipConnection(nn.Module):
    def __init__(self, summary_mode, att_mode, channels, num_layers):
        super(SummarizeSkipConnection, self).__init__()
        self.summary_mode = summary_mode
        self.att_mode = att_mode
        
        if self.summary_mode == 'lstm':
            out_channels = (num_layers * channels) // 2
        else: # if self.summary_mode == 'vanilla' or 'roll'
            out_channels = channels
        
        self.lstm = LSTM(channels, out_channels,
                             bidirectional=True, batch_first=True)
        # self.att = Linear(2 * out_channels, 1)
        self.weight_l = nn.Parameter(torch.empty((out_channels, 1), requires_grad=True))
        self.weight_r = nn.Parameter(torch.empty((out_channels, 1), requires_grad=True))
        self.kldiv = nn.KLDivLoss(reduction='none')

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        # self.att.reset_parameters()
        init.kaiming_uniform_(self.weight_l, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))

    def forward(self, hs, ws):
        h = torch.stack(hs, dim=1)  # h is (n, L, d).

        # 'Summary' takes h as input, query and key vector as output
        if self.summary_mode == 'vanilla':
            query = h.clone() # query's l-th row is h_i^l
            n_layers = h.size()[1]
            key = query[:, -1, :].repeat(n_layers, 1, 1).permute(1,0,2) # key's all row is h_i^L

        elif self.summary_mode == 'roll':
            query = h.clone() # query's l-th row is h_i^l
            key = torch.roll(h.clone(), -1, dims=1) # key's l-th row is h_i^(l+1)
            query, key, h = query[:, :-1, :], key[:, :-1, :], h[:, :-1, :] # dump last elements

        elif self.summary_mode == 'lstm':
            alpha, _ = self.lstm(h) # alpha (n, L, dL). dL/2 is hid_channels of forward or backward LSTM
            out_channels = alpha.size()[-1]
            query, key = alpha[:, :, :out_channels//2], alpha[:, :, out_channels//2:]


        # 'Attention' takes query and key as input, alpha as output
        if self.att_mode == 'dp':
            if self.summary_mode in ['vanilla', 'roll']:
                query = query_transform(query, ws, self.summary_mode)
            alpha = (query * key).sum(dim=-1) / math.sqrt(query.size()[-1])

        elif self.att_mode == 'ad':
            if self.summary_mode in ['vanilla', 'roll']:
                n_nodes = query.shape[0]
                weight_l, weight_r = weight_transform(self.weight_l, self.weight_r, ws, n_nodes, self.summary_mode)
                query = query_transform(query, ws, self.summary_mode)
            alpha = (weight_l*query + weight_r*key).sum(dim=-1)

        elif self.att_mode == 'mx': 
            query_key = torch.cat([query, key], dim=-1)
            alpha_ad = self.att(query_key).squeeze(-1)
            alpha = alpha_ad * torch.sigmoid((query * key).sum(dim=-1))

        alpha_softmax = torch.softmax(alpha, dim=-1)
        return alpha_softmax, (h * alpha_softmax.unsqueeze(-1)).sum(dim=1) # h_i = \sum_l alpha_i^l * h_i^l


# if cfg.skip_connection is in [vanilla, res, dense, highway]
class SkipConnection(nn.Module):
    def __init__(self, skip_connection, n_hidden):
        super(SkipConnection, self).__init__()
        self.skip_connection = skip_connection
        if self.skip_connection == 'highway':
            self.linear = Linear(n_hidden, n_hidden)

    def forward(self, h, x):
        if self.skip_connection == 'vanilla':
            return h

        elif self.skip_connection == 'res':
            return h + x

        elif self.skip_connection == 'dense':
            return torch.cat([h, x], dim=-1)
            
        elif self.skip_connection == 'highway':
            gating_weights = torch.sigmoid(self.linear(x))
            ones = torch.ones_like(gating_weights)
            return h*gating_weights + x*(ones-gating_weights) # h*W + x*(1-W)

def orthonomal_loss(model, device):
    def eyes_like(tensor): # eyes means identity matrix
        size = tensor.size()[0]
        return torch.eye(size, out=torch.empty_like(tensor)).to(device)

    def calc_orthonomal_loss(weight):
        mm = torch.mm(weight, torch.transpose(weight, 0, 1))
        return torch.norm(mm - eyes_like(mm))

    orthonomal_loss = torch.tensor(0, dtype=torch.float32).to(device)
    for conv in model.convs[1:]: # in [W^2, W^3, ... , W^L], not include W^1
        n_heads = 1
        if conv.name == 'gcn_conv':
            weights = [conv.conv.lin.weight]
        elif conv.name == 'gat_conv':
            n_heads = conv.conv.heads
            weights = [conv.conv.lin_src.weight]
        elif conv.name == 'sage_conv':
            weights = [conv.conv.lin_r.weight, conv.conv.lin_l.weight]

        for weight in weights:
            orthonomal_loss += (calc_orthonomal_loss(weight) / n_heads / len(weights))

    return orthonomal_loss