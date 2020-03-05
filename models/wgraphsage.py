import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph


class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, norm=None):
        super(NodeUpdate, self).__init__()
        self.fc_neigh = nn.Linear(in_features=in_feats, out_features=out_feats)
        self.activation = activation
        self.norm = norm
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, node):
        h_neigh = node.data['neigh']
        h_neigh = self.fc_neigh(h_neigh)
        if self.activation is not None:
            h_neigh = self.activation(h_neigh)
        if self.norm is not None:
            h_neigh = self.norm(h_neigh)
        return {'activation': h_neigh}


class WGraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=None, norm=None, dropout=0.0):
        super(WGraphSAGE, self).__init__()
        self.n_layers = n_layers
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()

        self.layers.append(NodeUpdate(in_feats=in_feats, out_feats=n_hidden, activation=activation, norm=norm))
        for _ in range(n_layers - 1):
            self.layers.append(NodeUpdate(in_feats=n_hidden, out_feats=n_hidden, activation=activation, norm=norm))
        self.linear = nn.Linear(n_hidden, n_classes)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def message_func(self, edges):
        return {'m': edges.src['h'] * edges.data['weight'], 'w': edges.data['weight']}
        # return {'m': edges.src['h']}

    def reduce_func(self, nodes):
        num_neigh = nodes.mailbox['m'].shape[1]  # number of neighbors
        neigh = (num_neigh * torch.sum(nodes.mailbox['m'], dim=1)) / (
                (num_neigh + 1) * torch.sum(nodes.mailbox['w'], dim=1)) + nodes.data['h'] / (num_neigh + 1)
        # neigh = (torch.sum(nodes.mailbox['m'], dim=1) + nodes.data['h']) / (num_neigh + 1)
        return {'neigh': neigh}

    def forward(self, nf: dgl.NodeFlow):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']
        nf.layers[1].data['h'] = nf.layers[1].data['features']
        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, self.message_func, self.reduce_func, layer)
        h = nf.layers[-1].data.pop('activation')
        h = self.linear(h)
        return h
