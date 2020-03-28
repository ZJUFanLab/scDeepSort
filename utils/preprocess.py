import argparse

import pandas as pd
import dgl
from time import time
import torch
import torch.nn.functional as F
import collections
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import pprint


def get_map_dict(species_data_path: Path, tissue):
    map_df = pd.read_excel(species_data_path / 'map.xlsx')
    # {num: {test_cell1: {train_cell1, train_cell2}, {test_cell2:....}}, num_2:{}...}
    map_dic = dict()
    for idx, row in enumerate(map_df.itertuples()):
        if getattr(row, 'Tissue') == tissue:
            num = getattr(row, 'num')
            test_celltype = getattr(row, 'Celltype')
            train_celltype = getattr(row, '_5')
            if map_dic.get(getattr(row, 'num')) is None:
                map_dic[num] = dict()
                map_dic[num][test_celltype] = set()
            elif map_dic[num].get(test_celltype) is None:
                map_dic[num][test_celltype] = set()
            map_dic[num][test_celltype].add(train_celltype)
    return map_dic


def normalize_weight(graph: dgl.DGLGraph):
    # normalize weight & add self-loop
    in_degrees = graph.in_degrees()
    for i in range(graph.number_of_nodes()):
        src, dst, in_edge_id = graph.in_edges(i, form='all')
        if src.shape[0] == 0:
            continue
        edge_w = graph.edata['weight'][in_edge_id]
        graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(edge_w)


def get_id_2_gene(gene_statistics_path, species_data_path, tissue, train_dir: str):
    if not gene_statistics_path.exists():
        data_path = species_data_path / train_dir
        data_files = data_path.glob(f'*{tissue}*_data.csv')
        genes = None
        for file in data_files:
            data = pd.read_csv(file, dtype=np.str, header=0).values[:, 0]
            if genes is None:
                genes = set(data)
            else:
                genes = genes | set(data)
        id2gene = list(genes)
        id2gene.sort()
        with open(gene_statistics_path, 'w', encoding='utf-8') as f:
            for gene in id2gene:
                f.write(gene + '\r\n')
    else:
        id2gene = []
        with open(gene_statistics_path, 'r', encoding='utf-8') as f:
            for line in f:
                id2gene.append(line.strip())
    return id2gene


def get_id_2_label(cell_statistics_path, species_data_path, tissue, train_dir: str):
    if not cell_statistics_path.exists():
        data_path = species_data_path / train_dir
        cell_files = data_path.glob(f'*{tissue}*_celltype.csv')
        cell_types = set()
        for file in cell_files:
            cell_types = set(pd.read_csv(file, dtype=np.str, header=0).values[:, 2]) | cell_types
        id2label = list(cell_types)
        with open(cell_statistics_path, 'w', encoding='utf-8') as f:
            for cell_type in id2label:
                f.write(cell_type + '\r\n')
    else:
        id2label = []
        with open(cell_statistics_path, 'r', encoding='utf-8') as f:
            for line in f:
                id2label.append(line.strip())
    return id2label


def calculate_mutual_info(gene_gene_edges: np.array, feat: np.array, gene_gene_graph: dgl.DGLGraph):
    """
    :param gene_gene_edges: (edges_num, 2)
    :param feat: (gene, cell)
    :param gene_gene_graph:
    :return:
    """
    feat = feat > 0
    edges_num = gene_gene_edges.shape[0]
    frequency = np.sum(feat, axis=1)  # get the frequency of every gene
    pair = feat[gene_gene_edges]  # (edges_num, 2, cell_num)
    pair_frequency = []  # f(x, y)
    for i in range(edges_num):
        pair_frequency.append(np.sum(pair[i, 0] & pair[i, 1]))
    pair_frequency = np.array(pair_frequency)
    # f(x, y) / (f(x) * f(y))
    mutual_info = pair_frequency / (frequency[gene_gene_edges[:, 0]] * frequency[gene_gene_edges[:, 1]] + 1e-6)
    mutual_info = F.relu(torch.log(torch.from_numpy(mutual_info)))
    mutual_info = torch.cat([mutual_info, mutual_info]).unsqueeze(-1)  # undirected edge
    return mutual_info


def load_data(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    train = params.train_dataset
    test = params.test_dataset
    tissue = params.tissue
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    species_data_path = proj_path / 'data' / params.species
    if not species_data_path.exists():
        raise NotImplementedError
    statistics_path = species_data_path / 'statistics'
    gene_gene_inter_path = species_data_path / 'interaction.csv'

    map_dict = get_map_dict(species_data_path, tissue)
    if not statistics_path.exists():
        statistics_path.mkdir()

    gene_statistics_path = statistics_path / (tissue + '_genes.txt')  # train+test gene
    cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')  # train labels

    # generate gene statistics file
    id2gene = get_id_2_gene(gene_statistics_path, species_data_path, tissue, params.train_dir)
    # generate cell label statistics file
    id2label = get_id_2_label(cell_statistics_path, species_data_path, tissue, params.train_dir)

    train_num, test_num = 0, 0
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f"totally {num_genes} genes, {num_labels} labels.")

    test_graph_dict = dict()  # test-graph dict
    test_label_dict = dict()  # test label dict
    test_index_dict = dict()  # test feature indices in all features
    test_mask_dict = dict()
    test_nid_dict = dict()

    if params.g2g:
        # 1. read data, store everything in a graph,
        gene_gene_edges = []  # gene-gene interaction edges
        gene_gene_inter = pd.read_csv(gene_gene_inter_path, index_col=0, header=0, dtype=np.str).values
        for g1, g2 in gene_gene_inter:
            g1_id = gene2id.get(g1, None)
            g2_id = gene2id.get(g2, None)
            if g1_id is not None and g2_id is not None:
                gene_gene_edges.append([g1_id, g2_id])
        gene_gene_edges = np.array(gene_gene_edges)
        gene_gene_num = gene_gene_edges.shape[0]  # number of gene-gene interactions
        print(f"totally {gene_gene_num} gene-gene edges")

    train_graph = dgl.DGLGraph()
    ids = torch.arange(num_genes, dtype=torch.int32, device=device).unsqueeze(-1)
    if params.g2g:
        w1 = torch.tensor([1] * gene_gene_num, dtype=torch.float32, device=device).unsqueeze(-1)
        w2 = torch.tensor([1] * gene_gene_num, dtype=torch.float32, device=device).unsqueeze(-1)

    '''
    CALCULATE GENE-GENE WEIGHT
    1. add gene-gene edges into train graph
    2. add gene-gene edges into gene-gene graph
    3. add gene-gene edges into test graph
    4. calculate weights of gene-gene edges in gene-gene graph
    5. copy weights of gene-gene edges from gene-gene graph to train&test graph
    '''
    # ==================================================
    # add all genes as nodes
    train_graph.add_nodes(num_genes, {'id': ids})
    if params.g2g:
        # add gene-gene edges
        train_graph.add_edges(gene_gene_edges[:, 0], gene_gene_edges[:, 1], {'weight': w1})
        train_graph.add_edges(gene_gene_edges[:, 1], gene_gene_edges[:, 0], {'weight': w2})

        gene_gene_graph = dgl.DGLGraph()  # a temporary graph used to calculate weight of gene-gene edge
        gene_gene_graph.add_nodes(num_genes)
        gene_gene_graph.add_edges(gene_gene_edges[:, 0], gene_gene_edges[:, 1])
        gene_gene_graph.add_edges(gene_gene_edges[:, 1], gene_gene_edges[:, 0])

    for num in test:
        test_graph_dict[num] = dgl.DGLGraph()
        test_graph_dict[num].add_nodes(num_genes, {'id': ids})
        if params.g2g:
            test_graph_dict[num].add_edges(gene_gene_edges[:, 0], gene_gene_edges[:, 1], {'weight': w1})
            test_graph_dict[num].add_edges(gene_gene_edges[:, 1], gene_gene_edges[:, 0], {'weight': w2})
    # ====================================================

    train_labels = []
    matrices = []
    start = time()
    for num in train + test:
        if num in train:
            data_path = species_data_path / (params.train_dir + f'/{params.species}_{tissue}{num}_data.csv')
            type_path = species_data_path / (params.train_dir + f'/{params.species}_{tissue}{num}_celltype.csv')
        else:
            data_path = species_data_path / (params.test_dir + f'/{params.species}_{tissue}{num}_data.csv')
            type_path = species_data_path / (params.test_dir + f'/{params.species}_{tissue}{num}_celltype.csv')

        # load celltype file then update labels accordingly
        cell2type = pd.read_csv(type_path, index_col=0)
        cell2type.columns = ['cell', 'type']
        if num in train:
            cell2type['id'] = cell2type['type'].map(label2id)
            assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
            train_labels += cell2type['id'].tolist()
        else:
            # test_labels += cell2type['type'].tolist()
            test_label_dict[num] = cell2type['type'].tolist()

        # load data file then update graph
        df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        df = df.transpose(copy=True)  # (cell, gene)

        assert cell2type['cell'].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(f'Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        if num in train:
            cell_idx = row_idx + train_graph.number_of_nodes()  # cell_index
        else:
            cell_idx = row_idx + test_graph_dict[num].number_of_nodes()
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)

        # test_nodes_index_dict[num] = list(range(graph.number_of_nodes(), graph.number_of_nodes() + len(df)))
        ids = torch.tensor([-1] * len(df), device=device, dtype=torch.int32).unsqueeze(-1)
        if num in train:
            train_num += len(df)
            train_graph.add_nodes(len(df), {'id': ids})
            train_graph.add_edges(cell_idx, gene_idx,
                                  {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})
            train_graph.add_edges(gene_idx, cell_idx,
                                  {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})
            for n in test:  # training cell also in test graph
                test_graph_dict[n].add_nodes(len(df), {'id': ids})
                test_graph_dict[n].add_edges(cell_idx, gene_idx,
                                             {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                                     device=device).unsqueeze(1)})
                test_graph_dict[n].add_edges(gene_idx, cell_idx,
                                             {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                                     device=device).unsqueeze(1)})
        else:
            test_index_dict[num] = list(range(train_num + test_num, train_num + test_num + len(df)))
            test_nid_dict[num] = list(
                range(test_graph_dict[num].number_of_nodes(), test_graph_dict[num].number_of_nodes() + len(df)))
            test_num += len(df)
            test_graph_dict[num].add_nodes(len(df), {'id': ids})
            # for the test cells, only gene-cell edges are in the test graph
            test_graph_dict[num].add_edges(gene_idx, cell_idx,
                                           {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                                   device=device).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(cell_idx)} edges.')
        print(f'#Nodes in Train Graph: {train_graph.number_of_nodes()}, #Edges: {train_graph.number_of_edges()}.')
        print(f'Costs {time() - start:.3f} s in total.\n')

    print(f"totally {train_num} nodes in train set, {test_num} nodes in test set.")

    train_index = list(range(num_genes + train_num))  # nodes index in train graph
    train_labels = list(map(int, train_labels))
    train_statistic = dict(collections.Counter(train_labels))
    print('------Train label statistics------')
    for i, (key, value) in enumerate(train_statistic.items(), start=1):
        print(f"#{i} [{id2label[key]}]: {value}")

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:train_num].T)
    gene_feat = gene_pca.transform(sparse_feat[:train_num].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float).to(device)
    train_graph.ndata['features'] = features[train_index]
    for num in test:
        test_graph_dict[num].ndata['features'] = features[train_index + test_index_dict[num]]
    # gene, train_cell, test_cell
    train_labels = torch.tensor([-1] * num_genes + train_labels, dtype=torch.long,
                                device=device)  # [gene_num+train_num]
    num_cells = test_num + train_num

    train_nid = torch.arange(start=num_genes, end=train_num + num_genes, dtype=torch.int64)

    # normalize weight
    normalize_weight(train_graph)
    # add self-loop
    train_graph.add_edges(train_graph.nodes(), train_graph.nodes(),
                          {'weight': torch.ones(train_graph.number_of_nodes(), dtype=torch.float,
                                                device=device).unsqueeze(1)})
    train_graph.readonly()

    for num in test:
        test_mask_dict[num] = torch.zeros(test_graph_dict[num].number_of_nodes(), dtype=torch.bool, device=device)
        test_mask_dict[num][test_nid_dict[num]] = 1
        test_nid_dict[num] = torch.tensor(test_nid_dict[num], dtype=torch.int64)
        # normalize weight & add self-loop
        normalize_weight(test_graph_dict[num])
        test_graph_dict[num].add_edges(test_graph_dict[num].nodes(), test_graph_dict[num].nodes(), {
            'weight': torch.ones(test_graph_dict[num].number_of_nodes(), dtype=torch.float, device=device).unsqueeze(
                1)})
        test_graph_dict[num].readonly()

    if params.g2g:
        '''
        CALCULATE GENE-GENE WEIGHT
        '''
        # normalize weight of gene-gene edges
        gene_gene_graph.edata['weight'] = calculate_mutual_info(gene_gene_edges, sparse_feat[:train_num].T,
                                                                gene_gene_graph)
        normalize_weight(gene_gene_graph)
        # copy weight of gene-gene edges from gene-graph to train_graph and test_graph
        assert gene_gene_num * 2 == gene_gene_graph.edata['weight'].shape[0]
        # NOTE: the first gene_gene_num*2 edges in graph are gene-gene interaction edges
        train_graph.edata['weight'][:gene_gene_num * 2] = gene_gene_graph.edata['weight'].to(device)
        for num in test:
            test_graph_dict[num].edata['weight'][:gene_gene_num * 2] = gene_gene_graph.edata['weight'].to(device)

    test_dict = {
        'graph': test_graph_dict,
        'label': test_label_dict,
        'nid': test_nid_dict,
        'mask': test_mask_dict
    }

    return num_cells, num_genes, num_labels, train_graph, train_labels, train_nid, map_dict, np.array(
        id2label, dtype=np.str), test_dict


if __name__ == '__main__':
    """
    python ./code/utils/preprocess.py --train_dataset 3285 753 --test_dataset 10100 19431 2502 2545 2695 3005 4397 --tissue Brain
    python ./code/utils/preprocess.py --train_dataset 4682 --test_dataset 203 2294 7701 8336 --tissue Kidney
    python ./code/utils/preprocess.py --train_dataset 2512 3014 1414 --test_dataset 1920 6340 707 769 --tissue Lung
    python ./code/utils/preprocess.py --train_dataset 4682 --test_dataset 203 --tissue Kidney
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--n_epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--dense_dim", type=int, default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="number of hidden gcn units")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--train_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--data_dir", type=str, default='mouse_data')
    parser.add_argument("--train_dir", type=str, default='mouse_train_data')
    parser.add_argument("--test_dir", type=str, default='mouse_test_data')
    parser.add_argument("--tissue", required=True, type=str)

    params = parser.parse_args()

    load_data(params)
