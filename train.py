import argparse
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.contrib.sampling import NeighborSampler
# self-defined
from utils import load_data_internal
from models import GNN
from pprint import pprint


class Trainer:
    def __init__(self, params):
        self.params = params
        self.prj_path = Path(__file__).parent.resolve()
        self.save_path = self.prj_path / 'pretrained' / f'{self.params.species}' / 'models'
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.num_cells, self.num_genes, self.num_labels, self.graph, self.train_ids, self.test_ids, self.labels = load_data_internal(params)
        self.labels = self.labels.to(self.device)
        self.model = GNN(in_feats=self.params.dense_dim,
                         n_hidden=self.params.hidden_dim,
                         n_classes=self.num_labels,
                         n_layers=self.params.n_layers,
                         gene_num=self.num_genes,
                         activation=F.relu,
                         dropout=self.params.dropout).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr,
                                          weight_decay=self.params.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        if self.params.num_neighbors == 0:
            self.num_neighbors = self.num_cells + self.num_genes
        else:
            self.num_neighbors = self.params.num_neighbors

        print(f"Train Number: {len(self.train_ids)}, Test Number: {len(self.test_ids)}")

    def fit(self):
        max_test_acc, _train_acc, _epoch = 0, 0, 0
        for epoch in range(self.params.n_epochs):
            loss = self.train()
            train_correct, train_unsure = self.evaluate(self.train_ids, 'train')
            train_acc = train_correct / len(self.train_ids)
            test_correct, test_unsure = self.evaluate(self.test_ids, 'test')
            test_acc = test_correct / len(self.test_ids)
            if max_test_acc <= test_acc:
                final_test_correct_num = test_correct
                final_test_unsure_num = test_unsure
                _train_acc = train_acc
                _epoch = epoch
                max_test_acc = test_acc
                self.save_model()
            print(
                f">>>>Epoch {epoch:04d}: Train Acc {train_acc:.4f}, Loss {loss / len(self.train_ids):.4f}, Test correct {test_correct}, "
                f"Test unsure {test_unsure}, Test Acc {test_acc:.4f}")
            if train_acc == 1:
                break

        print(f"---{self.params.species} {self.params.tissue} Best test result:---")
        print(f"Epoch {_epoch:04d}, Train Acc {_train_acc:.4f}, Test Correct Num {final_test_correct_num}, Test Total Num {len(self.test_ids)}, Test Unsure Num {final_test_unsure_num}, Test Acc {final_test_correct_num / len(self.test_ids):.4f}")

    def train(self):
        self.model.train()
        total_loss = 0
        for batch, nf in enumerate(NeighborSampler(g=self.graph,
                                                   batch_size=self.params.batch_size,
                                                   expand_factor=self.num_neighbors,
                                                   num_hops=self.params.n_layers,
                                                   neighbor_type='in',
                                                   shuffle=True,
                                                   num_workers=8,
                                                   seed_nodes=self.train_ids)):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            logits = self.model(nf)
            batch_nids = nf.layer_parent_nid(-1).type(torch.long).to(device=self.device)
            loss = self.loss_fn(logits, self.labels[batch_nids])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss

    def evaluate(self, ids, type='test'):
        self.model.eval()
        total_correct, total_unsure = 0, 0
        for nf in NeighborSampler(g=self.graph,
                                  batch_size=self.params.batch_size,
                                  expand_factor=self.num_cells + self.num_genes,
                                  num_hops=params.n_layers,
                                  neighbor_type='in',
                                  shuffle=True,
                                  num_workers=8,
                                  seed_nodes=ids):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            with torch.no_grad():
                logits = self.model(nf).cpu()
            batch_nids = nf.layer_parent_nid(-1).type(torch.long)
            logits = nn.functional.softmax(logits, dim=1).numpy()
            label_list = self.labels.cpu()[batch_nids]
            for pred, label in zip(logits, label_list):
                max_prob = pred.max().item()
                if max_prob < self.params.unsure_rate / self.num_labels:
                    total_unsure += 1
                elif pred.argmax().item() == label:
                    total_correct += 1

        return total_correct, total_unsure

    def save_model(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(state, self.save_path / f"{self.params.species}-{self.params.tissue}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=2,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--filetype", default='csv', type=str, choices=['csv', 'gz'],
                        help='data file type, csv or gz')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--dense_dim", type=int, default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="number of hidden gcn units")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--threshold", type=float, default=0,
                        help="the threshold to connect edges between cells and genes")
    parser.add_argument("--num_neighbors", type=int, default=0,
                        help="number of neighbors to sample in message passing process. 0 means all neighbors")
    parser.add_argument("--exclude_rate", type=float, default=0.005,
                        help="exclude some cells less than this rate.")
    parser.add_argument("--species", default='mouse', type=str)
    parser.add_argument("--tissue", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--unsure_rate", type=float, default=2.,
                        help="the threshold to predict unsure cell")
    parser.add_argument("--test_rate", type=float, default=0.2)

    params = parser.parse_args()
    pprint(vars(params))

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    trainer = Trainer(params)
    trainer.fit()
