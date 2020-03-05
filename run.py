import argparse
import time
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import dgl
from dgl.contrib.sampling import NeighborSampler
# self-defined
from utils import load_data
from models import GraphSAGE, WGraphSAGE


class Trainer:
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.num_cells, self.num_genes, self.num_classes, self.graph, self.features, self.train_labels, self.train_mask, self.train_index, self.train_nid, self.map_dict, self.id2label, self.test_dict = load_data(
            params)
        """
        test_dict = {'nodes_index': test_nodes_index_dict,
             'label': test_label_dict,
             'mask': test_masks_dict,
             'nid': test_nid_dict}
        """
        # self.model = GraphSAGE(in_feats=params.dense_dim,
        #                        n_hidden=params.hidden_dim,
        #                        n_classes=self.num_classes,
        #                        n_layers=params.n_layers,
        #                        activation=F.relu,
        #                        dropout=params.dropout,
        #                        aggregator_type=params.aggregator_type,
        #                        cell_w=params.cell_w,
        #                        gene_w=params.gene_w,
        #                        learned_w=params.learned_w)
        self.model = WGraphSAGE(in_feats=params.dense_dim,
                                n_hidden=params.hidden_dim,
                                n_classes=self.num_classes,
                                n_layers=params.n_layers,
                                activation=F.relu,
                                dropout=params.dropout)
        self.num_neighbors = self.graph.number_of_nodes()

        self.train_graph = self.graph.subgraph(self.train_index)
        self.train_graph.copy_from_parent()  # copy data from parent graph
        self.train_graph.readonly()

        self.test_graph = dict()
        for num in self.params.test_dataset:
            self.test_graph[num] = self.graph.subgraph(self.test_dict['nodes_index'][num])
            self.test_graph[num].copy_from_parent()
            self.test_graph[num].readonly()

        self.model.to(self.device)
        # self.features = self.features.to(self.device)
        self.train_mask = self.train_mask.to(self.device)

        for key in self.test_dict['mask']:
            self.test_dict['mask'][key] = self.test_dict['mask'][key].to(self.device)
            self.test_dict['nodes_index'][key] = self.test_dict['nodes_index'][key].to(self.device)
        self.train_labels = self.train_labels.to(self.device)

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=self.params.weight_decay)
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        final_record, max_mean_acc, final_train_acc = None, 0, 0
        for epoch in range(self.params.n_epochs):
            record = dict()
            total_train_correct, total_loss = 0, 0
            for nf in NeighborSampler(g=self.train_graph,
                                      batch_size=self.params.batch_size,
                                      expand_factor=self.num_neighbors,
                                      num_hops=params.n_layers,
                                      neighbor_type='in',
                                      shuffle=True,
                                      num_workers=8,
                                      seed_nodes=self.train_graph.map_to_subgraph_nid(self.train_nid)):
                nf.copy_from_parent()  # Copy node/edge features from the parent graph.
                logits = self.model(nf)
                # NOTE ids in sub-graph and parent graph are same
                batch_nids = nf.layer_parent_nid(-1).type(torch.long).to(device=self.device)
                loss = loss_fn(logits, self.train_labels[batch_nids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, indices = torch.max(logits, dim=1)
                total_train_correct += torch.sum(indices == self.train_labels[batch_nids]).item()
                total_loss += loss.item()

            # logits = self.model(self.train_graph, self.features[self.train_index])
            # loss = loss_fn(logits[self.train_mask], self.train_labels[self.train_mask])
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # _, _, train_acc = self.evaluate_train(self.train_mask)
            train_acc = total_train_correct / len(self.train_nid)
            test_total_acc = 0
            for num in self.params.test_dataset:
                c, t, test_acc = self.evaluate_test(num)
                record[num] = dict()
                record[num]['c'], record[num]['t'], record[num]['acc'] = c, t, test_acc
                test_total_acc += test_acc

            if test_total_acc / len(self.params.test_dataset) > max_mean_acc:
                max_mean_acc = test_total_acc / len(self.params.test_dataset)
                final_record = record
                final_train_acc = train_acc

            if epoch % 20 == 0:
                # print(f"Epoch {epoch:04d}: Acc {train_acc:.4f} / {test_acc:.4f}, Loss {loss:.4f}, [{c}/{t}]")
                print(f">>>>Epoch {epoch:04d}: Acc {train_acc:.4f}, Loss {total_loss / len(self.train_nid):.4f}")
                for num in self.params.test_dataset:
                    print(f"#{num} Test Acc: {record[num]['acc']:.4f}, [{record[num]['c']}/{record[num]['t']}]")
        print(f"---{self.params.tissue} Best test result:---")
        print(f"Train acc: {final_train_acc:.4f}")
        for num in self.params.test_dataset:
            print(
                f"#{num} Test Acc: {final_record[num]['acc']:.4f}, [{final_record[num]['c']}/{final_record[num]['t']}]")

    def evaluate_train(self, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.train_graph, self.features[self.train_index])
            logits = logits[mask]
            labels = self.train_labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels).item()
        total = mask.type(torch.LongTensor).sum().item()
        return correct, total, correct / total

    def evaluate_test(self, num):
        self.model.eval()
        new_logits = torch.zeros((self.num_genes + self.num_cells, self.num_classes))
        for nf in NeighborSampler(g=self.test_graph[num],
                                  batch_size=self.params.batch_size,
                                  expand_factor=self.num_neighbors,
                                  num_hops=self.params.n_layers,
                                  neighbor_type='in',
                                  shuffle=True,
                                  num_workers=8,
                                  seed_nodes=self.test_graph[num].map_to_subgraph_nid(self.test_dict['nid'][num])):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            with torch.no_grad():
                logits = self.model(nf).cpu()
            batch_nids = nf.layer_parent_nid(-1).type(torch.long)  # NOTE node ids in the subgraph
            batch_nids = self.test_graph[num].parent_nid[batch_nids]  # map ids from subgraph to parent graph
            new_logits[batch_nids] = logits
        # with torch.no_grad():
        #     logits = self.model(self.test_graph[num], self.features[self.test_dict['nodes_index'][num]]).cpu()
        # new_logits = torch.zeros((self.num_genes + self.num_cells, logits.shape[1]))
        # new_logits[parent_nid] = logits
        new_logits = new_logits[self.test_dict['mask'][num]]
        indices = new_logits.numpy().argmax(axis=1)
        predict_label = self.id2label[indices]
        correct = 0
        for p_label, t_label in zip(predict_label, self.test_dict['label'][num]):
            if p_label in self.map_dict[num][t_label]:
                correct += 1
        total = predict_label.shape[0]
        return correct, total, correct / total


if __name__ == '__main__':
    """
    python ./code/run.py --train_dataset 3285 753 --test_dataset 10100 19431 2502 2545 2695 3005 4397 --tissue Brain
    python ./code/run.py --train_dataset 3285 753 --test_dataset 19431 2695 2502 2545 3005 4397  --tissue Brain
    python ./code/run.py --train_dataset 4682 --test_dataset 203 2294 7701 8336 --tissue Kidney
    python ./code/run.py --train_dataset 4682 --test_dataset 203 2294 8336 --tissue Kidney
    python ./code/run.py --train_dataset 2512 3014 1414 --test_dataset 1920 6340 707 769 --tissue Lung
    python ./code/run.py --tissue Lung --train_dataset 2512 3014 1414 --test_dataset 769 707
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
    parser.add_argument("--n_epochs", type=int, default=300,
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
    parser.add_argument("--batch_size", type=int, default=1000)
    # parser.add_argument("--cell_w", type=float, default=1.)
    # parser.add_argument("--gene_w", type=float, default=1.)
    # parser.add_argument("--learned", dest='learned_w', action='store_true')
    # parser.add_argument("--no-learned", dest='learned_w', action='store_false')
    parser.set_defaults(learned_w=False)

    params = parser.parse_args()
    print(params)

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    trainer = Trainer(params)
    trainer.train()