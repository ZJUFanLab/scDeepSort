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
        if self.params.evaluate:
            self.total_cell, self.num_genes, self.num_classes, self.id2label, self.test_dict, self.map_dict, self.time = load_data(params)
        else:
            self.total_cell, self.num_genes, self.num_classes, self.id2label, self.test_dict, self.time = load_data(params)
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
            tic = time.time()
            if self.params.evaluate:
                correct, total, unsure, acc, pred = self.evaluate_test(num)
                print(f"{self.params.species}_{self.params.tissue} #{num} Test Acc: {acc:.4f} ({correct}/{total}), Number of Unsure Cells: {unsure}")
            else:
                pred = self.inference(num)
            toc = time.time()
            print(f'{self.params.species}_{self.params.tissue} #{num} Time Consumed: {toc - tic + self.time:.2f} seconds.')
            self.save_pred(num, pred)

    def load_model(self):
        model_path = self.prj_path / 'pretrained' / self.params.species / 'models' / f'{self.params.species}-{self.params.tissue}.pt'
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state['model'])

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
        new_logits = nn.functional.softmax(new_logits, dim=1).numpy()
        predict_label = []
        for pred in new_logits:
            pred_label = self.id2label[pred.argmax().item()]
            if pred.max().item() < self.params.unsure_rate / self.num_classes:
                # unsure
                predict_label.append('unsure')
            else:
                predict_label.append(pred_label)
        return predict_label

    def evaluate_test(self, num):
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
        new_logits = nn.functional.softmax(new_logits, dim=1).numpy()
        total = new_logits.shape[0]
        unsure_num, correct = 0, 0
        predict_label = []
        for pred, t_label in zip(new_logits, self.test_dict['label'][num]):
            pred_label = self.id2label[pred.argmax().item()]
            if pred.max().item() < self.params.unsure_rate / self.num_classes:
                # unsure
                unsure_num += 1
                predict_label.append('unsure')
            else:
                if pred_label in self.map_dict[num][t_label]:
                    correct += 1
                predict_label.append(pred_label)
        return correct, total, unsure_num, correct / total, predict_label

    def save_pred(self, num, pred):
        label_map = pd.read_excel('./map/celltype2subtype.xlsx',
                sheet_name=self.params.species, header=0,
                names=['species', 'old_type', 'new_type', 'new_subtype'])
        label_map = label_map.fillna('N/A', inplace=False)
        oldtype2newtype = {}
        oldtype2newsubtype = {}
        for _, old_type, new_type, new_subtype in label_map.itertuples(index=False):
            oldtype2newtype[old_type] = new_type
            oldtype2newsubtype[old_type] = new_subtype

        save_path = self.prj_path / self.params.save_dir
        if not save_path.exists():
            save_path.mkdir()
        if self.params.evaluate:
            df = pd.DataFrame({
                'index': self.test_dict['origin_id'][num],
                'original label': self.test_dict['label'][num],
                'cell_type': [oldtype2newtype.get(p, p) for p in pred],
                'cell_subtype': [oldtype2newsubtype.get(p, p) for p in pred]})
        else:
            df = pd.DataFrame({
                'index': self.test_dict['origin_id'][num],
                'cell_type': [oldtype2newtype.get(p, p) for p in pred],
                'cell_subtype': [oldtype2newsubtype.get(p, p) for p in pred]})
        df.to_csv(
            save_path / (self.params.species + f"_{self.params.tissue}_{num}.csv"),
            index=False)
        print(f"output has been stored in {self.params.species}_{self.params.tissue}_{num}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--filetype", default='csv', type=str, choices=['csv', 'gz'])
    parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--species", default='mouse', type=str, choices=['human', 'mouse'])
    parser.add_argument("--tissue", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--evaluate", dest='evaluate', action='store_true')
    parser.add_argument("--test", dest='evaluate', action='store_false')
    parser.add_argument("--unsure_rate", type=float, default=2.)
    parser.set_defaults(evaluate=True)
    params = parser.parse_args()
    params.dropout = 0.1
    params.dense_dim = 400
    params.hidden_dim = 200
    params.test_dir = 'test'
    params.random_seed = 10086
    params.threshold = 0
    params.save_dir = 'result'
    pprint(vars(params))

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    trainer = Runner(params)
    trainer.run()
