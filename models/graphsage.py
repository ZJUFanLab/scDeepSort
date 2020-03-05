import torch
from torch import nn
import torch.nn.functional as F
import dgl.function as fn


class SAGEConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None,
                 cell_w=1.,
                 gene_w=1.,
                 learned_w=False):
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        if learned_w:
            self.self_weight_cell = nn.Parameter(torch.tensor([cell_w], dtype=torch.float32))
            self.self_weight_gene = nn.Parameter(torch.tensor([gene_w], dtype=torch.float32))
        else:
            self.register_buffer('self_weight_cell', torch.tensor([cell_w], dtype=torch.float32))
            self.register_buffer('self_weight_gene', torch.tensor([gene_w], dtype=torch.float32))
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_feats, in_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m']  # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_feats)),
             m.new_zeros((1, batch_size, self._in_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def apply_node_func(self, nodes):
        self_w = torch.where(nodes.data['type'] == True, self.self_weight_cell, self.self_weight_gene)
        return {'result': nodes.data['neigh'] + self_w * nodes.data['h']}
        # return {'result': nodes.data['neigh'] + torch.cat(
        #     (self.self_weight_gene * nodes.data['h'][~nodes.data['type'].squeeze()],
        #      self.self_weight_cell * nodes.data['h'][nodes.data['type'].squeeze()]))}

    def forward(self, graph, feat):

        graph = graph.local_var()
        feat = self.feat_drop(feat)
        h_self = feat
        if self._aggre_type == 'mean':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'gcn':
            graph.ndata['h'] = feat
            # graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'), self.apply_node_func)
            # divide in_degrees
            # self_w = torch.where(graph.ndata['type'] == True, self.self_weight_cell.data, self.self_weight_gene.data)
            degs = graph.in_degrees().unsqueeze(-1).float().to(feat.device) + 1
            h_neigh = graph.ndata['result'] / degs
        elif self._aggre_type == 'pool':
            graph.ndata['h'] = F.relu(self.fc_pool(feat))
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'lstm':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            h_neigh = graph.ndata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            self_w = torch.where(graph.ndata['type'] == True, self.self_weight_cell, self.self_weight_gene)
            rst = self_w * self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 gene_w=1.,
                 cell_w=1.,
                 learned_w=False):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            SAGEConv(in_feats,
                     n_hidden,
                     aggregator_type,
                     feat_drop=dropout,
                     activation=activation,
                     gene_w=gene_w,
                     cell_w=cell_w,
                     learned_w=learned_w))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                SAGEConv(n_hidden,
                         n_hidden,
                         aggregator_type,
                         feat_drop=dropout,
                         activation=activation,
                         gene_w=gene_w,
                         cell_w=cell_w,
                         learned_w=learned_w))
        # output layer

        self.linear = nn.Linear(n_hidden, n_classes)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)  # (node, dim)
        h = self.linear(h)
        return h
