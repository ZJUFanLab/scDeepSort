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


def get_map_dict(mouse_data_path: Path, tissue):
    map_df = pd.read_excel(mouse_data_path / 'Cell_type_mapping.xlsx')
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


def load_data(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    train = params.train_dataset
    test = params.test_dataset
    tissue = params.tissue
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    mouse_data_path = proj_path / 'data' / 'mouse_data_v2'
    statistics_path = mouse_data_path / 'statistics'

    map_dict = get_map_dict(mouse_data_path, tissue)
    if not statistics_path.exists():
        statistics_path.mkdir()

    gene_statistics_path = statistics_path / (tissue + '_genes.txt')  # train+test gene
    cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')  # train labels

    # generate gene statistics file
    if not gene_statistics_path.exists():
        data_files = mouse_data_path.glob(f'**/*{tissue}*_data.csv')
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

    # generate cell label statistics file
    if not cell_statistics_path.exists():
        cell_files = mouse_data_path.glob(f'mouse_training_data/*{tissue}*_celltype.csv')
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

    train_num, test_num = 0, 0
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f"totally {num_genes} genes, {num_labels} labels.")

    # 1. read data, restore everything in a graph,
    graph = dgl.DGLGraph()
    start = time()
    # add all genes as nodes
    graph.add_nodes(num_genes)
    train_labels = []
    test_nodes_index_dict = dict()  # {num: [idx1, idx2...], ...}
    test_label_dict = dict()  # {num: [label1, label2...], ...}
    test_masks_dict = dict()  # {num: test_mask, ...}
    test_nid_dict = dict()  # {num: test_nid}
    matrices = []
    for num in train + test:
        if num in train:
            data_path = mouse_data_path / (params.train_dir + f'mouse_{tissue}{num}_data.csv')
            type_path = mouse_data_path / (params.train_dir + f'mouse_{tissue}{num}_celltype.csv')
        else:
            data_path = mouse_data_path / (params.test_dir + f'mouse_{tissue}{num}_data.csv')
            type_path = mouse_data_path / (params.test_dir + f'mouse_{tissue}{num}_celltype.csv')

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
        # print(df.head())
        print(f'Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        src_idx = row_idx + graph.number_of_nodes()  # cell_index
        tgt_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, tgt_idx)), shape=info_shape)
        matrices.append(info)

        if num in train:
            train_num += len(df)
        else:
            test_num += len(df)
            test_nodes_index_dict[num] = list(range(graph.number_of_nodes(), graph.number_of_nodes() + len(df)))

        graph.add_nodes(len(df))
        # graph.add_edges(src_idx, tgt_idx)
        # graph.add_edges(tgt_idx, src_idx)
        graph.add_edges(src_idx, tgt_idx,
                        {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})
        graph.add_edges(tgt_idx, src_idx,
                        {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(src_idx)} edges.')
        print(f'#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')
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

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)
    # gene, train_cell, test_cell
    train_labels = torch.tensor([-1] * num_genes + train_labels, dtype=torch.long)
    num_cells = test_num + train_num

    train_mask = torch.zeros(train_num + num_genes, dtype=torch.bool)
    train_mask[num_genes:train_num + num_genes] += 1  # indices of trained nodes
    train_nid = torch.arange(start=num_genes, end=train_num + num_genes, dtype=torch.int64)

    for num in test:
        test_nid_dict[num] = torch.tensor(test_nodes_index_dict[num], dtype=torch.int64)
        test_masks_dict[num] = torch.zeros(num_cells + num_genes, dtype=torch.bool)
        test_masks_dict[num][test_nodes_index_dict[num]] = 1
        # nodes in every test subgraph
        test_nodes_index_dict[num] = torch.tensor(test_nodes_index_dict[num] + train_index)

    graph.ndata['features'] = features.to(device)
    test_dict = {'nodes_index': test_nodes_index_dict,  # nodes in every test subgraph
                 'label': test_label_dict,
                 'mask': test_masks_dict,  # indices of tested nodes
                 'nid': test_nid_dict
                 }
    # assert train_mask.sum().item() + test_mask.sum().item() == num_cells
    return num_cells, num_genes, num_labels, graph, features, train_labels, train_mask, train_index, train_nid, map_dict, np.array(
        id2label, dtype=np.str), test_dict


if __name__ == '__main__':
    """
    python ./code/datasets/mouse.py --train_dataset 3510 --test_dataset 1059 --tissue Mammary_gland
    python ./code/datasets/mouse.py --train_dataset 3510 1311 6633 6905 4909 2081 --test_dataset 1059 648 1592 --tissue Mammary_gland --exclude Neuron Mast.cell Luminal.progenitor.cell
    python ./code/datasets/mouse.py --train_dataset 3285 753 --test_dataset 10100 19431 2502 2545 2695 3005 4397 --tissue Brain
    python ./code/datasets/mouse.py --train_dataset 4682 --test_dataset 203 2294 7701 8336 --tissue Kidney
    python ./code/datasets/preprocess.py --train_dataset 2512 3014 1414 --test_dataset 1920 6340 707 769 --tissue Lung
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
    parser.add_argument("--aggregator_type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--train_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--train_dir", type=str, default='mouse_training_data/')
    parser.add_argument("--test_dir", type=str, default='mouse_test_data/')
    parser.add_argument("--tissue", required=True, type=str,
                        help="list of dataset id")

    params = parser.parse_args()

    load_data(params)
