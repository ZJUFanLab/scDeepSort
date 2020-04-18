import argparse
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import dgl
from dgl.contrib.sampling import NeighborSampler
# self-defined
from utils import load_data, get_logger
from models import GraphSAGE, WGraphSAGE
from pprint import pprint


class Trainer:
    def __init__(self, params):
        self.params = params
        self.postfix = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.logger = get_logger(Path(__file__).parent.resolve().parent.resolve() / self.params.log_dir,
                                 self.params.log_file)
        self.num_cells, self.num_genes, self.num_classes, self.train_graph, self.train_labels, self.train_nid, self.map_dict, self.id2label, self.test_dict = load_data(
            params)
        """
        test_dict = {
            'graph': test_graph_dict,
            'label': test_label_dict,
            'nid': test_index_dict,
            'mask': test_mask_dict
        """

        self.model = WGraphSAGE(in_feats=params.dense_dim,
                                n_hidden=params.hidden_dim,
                                n_classes=self.num_classes,
                                n_layers=params.n_layers,
                                gene_num=self.num_genes,
                                activation=F.relu,
                                dropout=params.dropout)
        if self.params.num_neighbors == 0:
            self.num_neighbors = self.num_cells + self.num_genes
        else:
            self.num_neighbors = self.params.num_neighbors
        self.model.to(self.device)
        self.final_record = dict()
        for num in self.params.test_dataset:
            self.final_record[num] = dict()
            self.final_record[num]['acc'] = 0

        self.logger.info("========================================")
        self.logger.info(vars(self.params))

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=self.params.weight_decay)
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        for epoch in range(self.params.n_epochs):
            tmp_record = dict()
            total_loss = 0
            for batch, nf in enumerate(NeighborSampler(g=self.train_graph,
                                                       batch_size=self.params.batch_size,
                                                       expand_factor=self.num_neighbors,
                                                       num_hops=params.n_layers,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=8,
                                                       seed_nodes=self.train_nid,
                                                       transition_prob=self.train_graph.edata['weight'].view(
                                                           -1).cpu())):
                if not self.params.evaluate_on_gpu:
                    self.model.to(self.device)
                self.model.train()
                nf.copy_from_parent()  # Copy node/edge features from the parent graph.
                logits = self.model(nf)
                batch_nids = nf.layer_parent_nid(-1).type(torch.long).to(device=self.device)
                loss = loss_fn(logits, self.train_labels[batch_nids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # ==============================================
                # if batch % 10 == 0:
                #     train_acc = self.evaluate_train() / len(self.train_nid)
                #     print(f"[E: {epoch:04d}| B: {batch:04d}] , Acc {train_acc:.4f}")
                #     for num in self.params.test_dataset:
                #         c, t, test_acc, pred = self.evaluate_test(num)
                #         if test_acc >= self.final_record[num]['acc']:
                #             self.final_record[num]['acc'] = test_acc
                #             self.final_record[num]['c'] = c
                #             self.final_record[num]['t'] = t
                #             self.final_record[num]['train_acc'] = train_acc
                #             self.final_record[num]['pred'] = pred
                #         print(
                #             f"#{num} Test Acc: {test_acc:.4f}, [{c}/{t}]")
                # ================================================

            train_acc = self.evaluate_train() / len(self.train_nid)
            # ================================================
            for num in self.params.test_dataset:
                c, t, test_acc, pred = self.evaluate_test(num)
                if test_acc >= self.final_record[num]['acc']:
                    self.final_record[num]['acc'] = test_acc
                    self.final_record[num]['c'] = c
                    self.final_record[num]['t'] = t
                    self.final_record[num]['train_acc'] = train_acc
                    self.final_record[num]['pred'] = pred
                tmp_record[num] = dict()
                tmp_record[num]['c'], tmp_record[num]['t'], tmp_record[num]['acc'] = c, t, test_acc
            print(f">>>>Epoch {epoch:04d}: Acc {train_acc:.4f}, Loss {total_loss / len(self.train_nid):.4f}")
            for num in self.params.test_dataset:
                print(
                    f"#{num} Test Acc: {tmp_record[num]['acc']:.4f}, [{tmp_record[num]['c']}/{tmp_record[num]['t']}]")
            # ================================================

            if train_acc == 1:
                break
        self.save_pred()
        print(f"---{self.params.tissue} Best test result:---")
        self.logger.info(f"---{self.params.tissue} Best test result:---")
        for num in self.params.test_dataset:
            print(
                f"#{num} Train Acc: {self.final_record[num]['train_acc']:.4f}, Test Acc: {self.final_record[num]['acc']:.4f}, [{self.final_record[num]['c']}/{self.final_record[num]['t']}]")
            self.logger.info(
                f"#{num} Train Acc: {self.final_record[num]['train_acc']:.4f}, Test Acc: {self.final_record[num]['acc']:.4f}, [{self.final_record[num]['c']}/{self.final_record[num]['t']}]")

    def evaluate_train(self):
        self.model.eval()
        if not self.params.evaluate_on_gpu:
            self.model.cpu()
        total_train_correct = 0
        for nf in NeighborSampler(g=self.train_graph,
                                  batch_size=self.params.batch_size,
                                  expand_factor=self.num_cells + self.num_genes,
                                  num_hops=params.n_layers,
                                  neighbor_type='in',
                                  shuffle=True,
                                  num_workers=8,
                                  seed_nodes=self.train_nid):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            with torch.no_grad():
                if self.params.evaluate_on_gpu:
                    logits = self.model(nf)
                else:
                    logits = self.model.evaluate(nf)
            batch_nids = nf.layer_parent_nid(-1).type(torch.long)
            _, indices = torch.max(logits, dim=1)
            if self.params.evaluate_on_gpu:
                total_train_correct += torch.sum(indices == self.train_labels[batch_nids]).item()
            else:
                total_train_correct += torch.sum(indices == self.train_labels[batch_nids].cpu()).item()
        return total_train_correct

    def evaluate_test(self, num):
        if not self.params.evaluate_on_gpu:
            self.model.cpu()
        self.model.eval()
        new_logits = torch.zeros((self.test_dict['graph'][num].number_of_nodes(), self.num_classes))
        for nf in NeighborSampler(g=self.test_dict['graph'][num],
                                  batch_size=self.params.batch_size,
                                  expand_factor=self.num_cells + self.num_genes,
                                  num_hops=self.params.n_layers,
                                  neighbor_type='in',
                                  shuffle=True,
                                  num_workers=8,
                                  seed_nodes=self.test_dict['nid'][num]):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            with torch.no_grad():
                if self.params.evaluate_on_gpu:
                    logits = self.model(nf).cpu()
                else:
                    logits = self.model.evaluate(nf)
            batch_nids = nf.layer_parent_nid(-1).type(torch.long)
            # batch_nids = self.test_graph[num].parent_nid[batch_nids]  # map ids from subgraph to parent graph
            new_logits[batch_nids] = logits

        new_logits = new_logits[self.test_dict['mask'][num]]
        indices = new_logits.numpy().argmax(axis=1)
        predict_label = self.id2label[indices]
        correct = 0
        for p_label, t_label in zip(predict_label, self.test_dict['label'][num]):
            if p_label in self.map_dict[num][t_label]:
                correct += 1
        total = predict_label.shape[0]
        return correct, total, correct / total, predict_label

    def save_pred(self):
        proj_path = Path(__file__).parent.resolve().parent.resolve()
        save_path = proj_path / self.params.save_dir
        if not save_path.exists():
            save_path.mkdir()
        for num in self.params.test_dataset:
            df = pd.DataFrame({'original label': self.test_dict['label'][num],
                               'prediction': self.final_record[num]['pred']})
            df.to_csv(
                save_path / (self.params.species + f"_{self.params.tissue}_{num}_" + self.postfix + ".csv"),
                index=False)
            # np.savetxt(save_path / f"{self.params.tissue}_{num}.txt", self.final_record[num]['pred'], fmt="%s",
            #            delimiter="\n")


if __name__ == '__main__':
    """
    
    python ./code/run.py --tissue Blood --train_dataset 283 2466 3201 135 352 658 --test_dataset 768 1109 1223 1610
    python ./code/run.py --tissue Bone_marrow --train_dataset 510 5298 13019 8166 --test_dataset 47 467
    python ./code/run.py --tissue Brain --train_dataset 3285 753 --test_dataset 19431 2695 2502 2545 3005 4397
    python ./code/run.py --tissue Fetal_brain --train_dataset 4369 --test_dataset 369
    python ./code/run.py --tissue Intestine --train_dataset 3438 1671 1575 --test_dataset 28 192 260 3260 1449 529
    python ./code/run.py --tissue Kidney --train_dataset 4682 --test_dataset 203 7926 1435
    python ./code/run.py --tissue Liver --train_dataset 4424 261 --test_dataset 3729 4122 7761 18000
    python ./code/run.py --tissue Lung --train_dataset 2512 1414 3014 --test_dataset 769 707 1920 6340
    python ./code/run.py --tissue Mammary_gland --train_dataset 3510 1311 6633 6905 4909 2081 1059 648 1592 --test_dataset 133
    python ./code/run.py --tissue Pancreas --train_dataset 3610 --test_dataset 1354 108 131 207
    python ./code/run.py --tissue Spleen --train_dataset 1970 --test_dataset 1759 1433 1081
    python ./code/run.py --tissue Testis --train_dataset 2216 11789 --test_dataset 2584 4239 8792 9923 6598 300 4233 1662 299 199 398 296 --batch_size 200
    --gpu "${gpu_id}" --dropout "${dropout}" --random_seed "${random_seed}" --log_file "${log_file}"
    --gpu "${gpu_id}" --log_file "${log_file}"
    
    python ./code/run.py --species human --tissue Blood --train_dataset 2719 5296 2156 7160 --test_dataset 9649 3223 2469 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Brain --train_dataset 7324 --test_dataset 251 2892 1834 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Colorectum --train_dataset 4681 3367 5549 5718 3281 5765 11229 --test_dataset 94 11894 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Esophagus --train_dataset 2696 8668 --test_dataset 16999 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Esophagus --train_dataset 2696 8668 --test_dataset 17001 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Esophagus --train_dataset 2696 8668 --test_dataset 16469 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Fetal_kidney --train_dataset 4734 9932 3057 --test_dataset 540 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Kidney --train_dataset 9153 9966 3849 --test_dataset 5675 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Liver --train_dataset 1811 4377 4384 --test_dataset 3502 298 5105 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Lung --train_dataset 6022 9603 --test_dataset 2064 6338 7211 10743 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Lung --train_dataset 6022 9603 --test_dataset 11204 4624 2841 9566 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Pancreas --train_dataset 9727 --test_dataset 465 958 20 185 15 11 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Placenta --train_dataset 9595 --test_dataset 615 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Spleen --train_dataset 15806 --test_dataset 11081 9887 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Spleen --train_dataset 15806 --test_dataset 18513 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Spleen --train_dataset 15806 --test_dataset 16286 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human --tissue Spleen --train_dataset 15806 --test_dataset 14848 --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run.py --species human
    python ./code/run.py --species human
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout", type=float, default=0.1,
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
    parser.add_argument("--num_neighbors", type=int, default=0)
    parser.add_argument("--train_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--g2g", dest='g2g', action='store_true')
    parser.add_argument("--no-g2g", dest='g2g', action='store_false')
    parser.add_argument("--species", default='mouse', type=str)
    parser.add_argument("--tissue", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--log_file", type=str, default='log')
    # parser.add_argument("--data_dir", type=str, default='mouse_data')
    parser.add_argument("--train_dir", type=str, default='train')
    parser.add_argument("--test_dir", type=str, default='test')
    parser.add_argument("--save_dir", type=str, default='result')
    parser.add_argument("--evaluate-on-gpu", dest='evaluate_on_gpu', action='store_true')
    parser.add_argument("--evaluate-on-cpu", dest='evaluate_on_gpu', action='store_false')
    parser.set_defaults(g2g=False)
    parser.set_defaults(evaluate_on_gpu=True)
    params = parser.parse_args()
    pprint(vars(params))

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    trainer = Trainer(params)
    trainer.train()
