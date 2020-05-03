import argparse
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.contrib.sampling import NeighborSampler
# self-defined
from utils import load_data
from models import GNN
from pprint import pprint


class Runner:
    def __init__(self, params):
        self.params = params
        self.postfix = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        self.prj_path = Path(__file__).parent.resolve()
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.total_cell, self.num_genes, self.num_classes, self.id2label, self.test_dict = load_data(params)
        """
        test_dict = {
            'graph': test_graph_dict,
            'nid': test_index_dict,
            'mask': test_mask_dict
        """
        self.model = GNN(in_feats=params.dense_dim,
                         n_hidden=params.hidden_dim,
                         n_classes=self.num_classes,
                         n_layers=1,
                         gene_num=self.num_genes,
                         activation=F.relu,
                         dropout=params.dropout)
        self.load_model()
        self.num_neighbors = self.total_cell + self.num_genes
        self.model.to(self.device)

    def run(self):
        for num in self.params.test_dataset:
            pred = self.inference(num)
            self.save_pred(num, pred)

    def load_model(self):
        model_path = self.prj_path / 'pretrained' / self.params.species / 'models' / f'{self.params.species}-{self.params.tissue}.pt'
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        # self.optimizer.load_state_dict(state['optimizer'])

    def inference(self, num):
        self.model.eval()
        new_logits = torch.zeros((self.test_dict['graph'][num].number_of_nodes(), self.num_classes))
        for nf in NeighborSampler(g=self.test_dict['graph'][num],
                                  batch_size=self.params.batch_size,
                                  expand_factor=self.total_cell + self.num_genes,
                                  num_hops=1,
                                  neighbor_type='in',
                                  shuffle=False,
                                  num_workers=8,
                                  seed_nodes=self.test_dict['nid'][num]):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            with torch.no_grad():
                logits = self.model(nf).cpu()
            batch_nids = nf.layer_parent_nid(-1).type(torch.long)
            new_logits[batch_nids] = logits

        new_logits = new_logits[self.test_dict['mask'][num]]
        indices = new_logits.numpy().argmax(axis=1)
        predict_label = self.id2label[indices]
        return predict_label


    def save_pred(self, num, pred):
        save_path = self.prj_path / self.params.save_dir
        if not save_path.exists():
            save_path.mkdir()
        df = pd.DataFrame({'prediction': pred})
        df.to_csv(
            save_path / (self.params.species + f"_{self.params.tissue}_{num}.csv"),
            index=False)


if __name__ == '__main__':
    """
    python run.py --species mouse --tissue Bone_marrow --test_dataset 47 --save_dir out
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--species", default='mouse', type=str)
    parser.add_argument("--tissue", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default='result')
    params = parser.parse_args()
    params.dropout = 0.1
    params.dense_dim = 400
    params.hidden_dim = 200
    params.test_dir = 'test'
    params.random_seed = 10086
    pprint(vars(params))

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    trainer = Runner(params)
    trainer.run()
