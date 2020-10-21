import argparse
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
import collections
from scipy.sparse import csr_matrix, vstack, load_npz
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
from time import time


def get_map_dict(map_path: Path, tissue):
    map_df = pd.read_excel(map_path / 'map.xlsx')
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


def get_id_2_gene(gene_statistics_path):
    id2gene = []
    with open(gene_statistics_path, 'r', encoding='utf-8') as f:
        for line in f:
            id2gene.append(line.strip())
    return id2gene


def get_id_2_label(cell_statistics_path):
    id2label = []
    with open(cell_statistics_path, 'r', encoding='utf-8') as f:
        for line in f:
            id2label.append(line.strip())
    return id2label


def load_data(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    test = params.test_dataset
    tissue = params.tissue
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    proj_path = Path(__file__).parent.resolve().parent.resolve()
    species_data_path = proj_path / 'pretrained' / params.species
    statistics_path = species_data_path / 'statistics'

    if params.evaluate:
        map_path = proj_path / 'map' / params.species
        map_dict = get_map_dict(map_path, tissue)

    if not statistics_path.exists():
        statistics_path.mkdir()

    gene_statistics_path = statistics_path / (tissue + '_genes.txt')  # train+test gene
    cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')  # train labels

    # generate gene statistics file
    id2gene = get_id_2_gene(gene_statistics_path)
    # generate cell label statistics file
    id2label = get_id_2_label(cell_statistics_path)

    test_num = 0
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f"The build graph contains {num_genes} gene nodes with {num_labels} labels supported.")

    test_graph_dict = dict()  # test-graph dict
    if params.evaluate:
        test_label_dict = dict()  # test label dict
    test_index_dict = dict()  # test feature indices in all features
    test_mask_dict = dict()
    test_nid_dict = dict()
    test_cell_origin_id_dict = dict()

    ids = torch.arange(num_genes, dtype=torch.int32, device=device).unsqueeze(-1)

    # ==================================================
    # add all genes as nodes

    for num in test:
        test_graph_dict[num] = dgl.DGLGraph()
        test_graph_dict[num].add_nodes(num_genes, {'id': ids})
    # ====================================================

    matrices = []

    support_data = proj_path / 'pretrained' / f'{params.species}' / 'graphs' / f'{params.species}_{tissue}_data.npz'
    support_num = 0
    info = load_npz(support_data)
    print(f"load {support_data.name}")
    row_idx, gene_idx = np.nonzero(info > 0)
    non_zeros = info.data
    cell_num = info.shape[0]
    support_num += cell_num
    matrices.append(info)
    ids = torch.tensor([-1] * cell_num, device=device, dtype=torch.int32).unsqueeze(-1)
    total_cell = support_num

    for n in test:  # training cell also in test graph
        cell_idx = row_idx + test_graph_dict[n].number_of_nodes()
        test_graph_dict[n].add_nodes(cell_num, {'id': ids})
        test_graph_dict[n].add_edges(cell_idx, gene_idx,
                                     {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                             device=device).unsqueeze(1)})
        test_graph_dict[n].add_edges(gene_idx, cell_idx,
                                     {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                             device=device).unsqueeze(1)})

    for num in test:
        data_path = proj_path / params.test_dir / params.species / f'{params.species}_{tissue}{num}_data.{params.filetype}'
        if params.evaluate:
            type_path = proj_path / params.test_dir / params.species / f'{params.species}_{tissue}{num}_celltype.csv'
            # load celltype file then update labels accordingly
            cell2type = pd.read_csv(type_path, index_col=0)
            cell2type.columns = ['cell', 'type']
            cell2type['type'] = cell2type['type'].map(str.strip)
            # test_labels += cell2type['type'].tolist()
            test_label_dict[num] = cell2type['type'].tolist()

        # load data file then update graph
        if params.filetype == 'csv':
            df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        elif params.filetype == 'gz':
            df = pd.read_csv(data_path, compression='gzip', index_col=0)
        else:
            print(f'Not supported type for {data_path}. Please verify your data file')

        test_cell_origin_id_dict[num] = list(df.columns)
        df = df.transpose(copy=True)  # (cell, gene)

        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(f'{params.species}_{tissue}{num}_data.{params.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')
        tic = time()
        print(f'Begin to cumulate time of training/testing ...')
        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        cell_idx = row_idx + test_graph_dict[num].number_of_nodes()
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)

        # test_nodes_index_dict[num] = list(range(graph.number_of_nodes(), graph.number_of_nodes() + len(df)))
        ids = torch.tensor([-1] * len(df), device=device, dtype=torch.int32).unsqueeze(-1)
        test_index_dict[num] = list(range(support_num + test_num, support_num + test_num + len(df)))
        test_nid_dict[num] = list(
            range(test_graph_dict[num].number_of_nodes(), test_graph_dict[num].number_of_nodes() + len(df)))
        test_num += len(df)
        test_graph_dict[num].add_nodes(len(df), {'id': ids})
        # for the test cells, only gene-cell edges are in the test graph
        test_graph_dict[num].add_edges(gene_idx, cell_idx,
                                       {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                               device=device).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(cell_idx)} edges.')
        total_cell += num

    support_index = list(range(num_genes + support_num))
    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:support_num].T)
    gene_feat = gene_pca.transform(sparse_feat[:support_num].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float).to(device)
    for num in test:
        test_graph_dict[num].ndata['features'] = features[support_index + test_index_dict[num]]

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

    if params.evaluate:
        test_dict = {
            'graph': test_graph_dict,
            'label': test_label_dict,
            'nid': test_nid_dict,
            'mask': test_mask_dict,
            'origin_id': test_cell_origin_id_dict
        }
        time_used = time() - tic
        return total_cell, num_genes, num_labels, np.array(id2label, dtype=np.str), test_dict, map_dict, time_used
    else:
        test_dict = {
            'graph': test_graph_dict,
            'nid': test_nid_dict,
            'mask': test_mask_dict,
            'origin_id': test_cell_origin_id_dict
        }
        time_used = time() - tic
        return total_cell, num_genes, num_labels, np.array(id2label, dtype=np.str), test_dict, time_used
