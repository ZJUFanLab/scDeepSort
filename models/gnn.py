import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class GNN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, gene_num, activation=None, norm=None, dropout=0.0):
        super(GNN, self).__init__()
        self.n_layers = n_layers
        self.gene_num = gene_num
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        self.layers.append(NodeUpdate(in_feats=in_feats, out_feats=n_hidden, activation=activation, norm=norm))
        for _ in range(n_layers - 1):
            self.layers.append(NodeUpdate(in_feats=n_hidden, out_feats=n_hidden, activation=activation, norm=norm))

        # [gene_num] is alpha of gene-gene, [gene_num+1] is alpha of cell-cell self loop
        self.alpha = nn.Parameter(torch.tensor([1] * (self.gene_num + 2), dtype=torch.float32).unsqueeze(-1))
        self.linear = nn.Linear(n_hidden, n_classes)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def message_func(self, edges: dgl.EdgeBatch):
        number_of_edges = edges.src['h'].shape[0]
        indices = np.expand_dims(np.array([self.gene_num + 1] * number_of_edges, dtype=np.int32), axis=1)
        src_id, dst_id = edges.src['id'].cpu().numpy(), edges.dst['id'].cpu().numpy()
        indices = np.where((src_id >= 0) & (dst_id < 0), src_id, indices)  # gene->cell
        indices = np.where((dst_id >= 0) & (src_id < 0), dst_id, indices)  # cell->gene
        indices = np.where((dst_id >= 0) & (src_id >= 0), self.gene_num, indices)  # gene-gene
        h = edges.src['h'] * self.alpha[indices.squeeze()]
        # return {'m': h}
        return {'m': h * edges.data['weight']}

    def forward(self, nf: dgl.NodeFlow):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']
        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, self.message_func, fn.mean('m', 'neigh'), layer)
        h = nf.layers[-1].data.pop('activation')
        h = self.linear(h)
        return h

    def evaluate(self, nf: dgl.NodeFlow):
        def message_func(edges: dgl.EdgeBatch):
            # edges.src['h']： (number of edges, feature dim)
            number_of_edges = edges.src['h'].shape[0]
            indices = np.expand_dims(np.array([self.gene_num + 1] * number_of_edges, dtype=np.int32), axis=1)
            src_id, dst_id = edges.src['id'].cpu().numpy(), edges.dst['id'].cpu().numpy()
            indices = np.where((src_id >= 0) & (dst_id < 0), src_id, indices)  # gene->cell
            indices = np.where((dst_id >= 0) & (src_id < 0), dst_id, indices)  # cell->gene
            indices = np.where((dst_id >= 0) & (src_id >= 0), self.gene_num, indices)  # gene-gene
            h = edges.src['h'].cpu() * self.alpha[indices.squeeze()]
            return {'m': h * edges.data['weight'].cpu()}

        nf.layers[0].data['activation'] = nf.layers[0].data['features'].cpu()
        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, message_func, fn.mean('m', 'neigh'), layer)
        h = nf.layers[-1].data.pop('activation')
        h = self.linear(h)
        return h
