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
from utils import load_data_atlas, get_logger
from models import GraphSAGE, WGraphSAGE
from pprint import pprint


class Trainer:
    def __init__(self, params):
        self.params = params
        self.postfix = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.logger = get_logger(Path(__file__).parent.resolve().parent.resolve() / self.params.log_dir,
                                 self.params.log_file)
        self.logger.info("========================================")
        self.logger.info(vars(self.params))
        self.num_cells, self.num_genes, self.num_labels, self.graph, self.train_ids, self.test_ids, self.labels = load_data_atlas(
            params, self.logger)
        self.model = WGraphSAGE(in_feats=params.dense_dim,
                                n_hidden=params.hidden_dim,
                                n_classes=self.num_labels,
                                n_layers=params.n_layers,
                                gene_num=self.num_genes,
                                activation=F.relu,
                                dropout=params.dropout).to(self.device)
        if self.params.num_neighbors == 0:
            self.num_neighbors = self.num_cells + self.num_genes
        else:
            self.num_neighbors = self.params.num_neighbors
        print(f"Train Number: {len(self.train_ids)}, Test Number: {len(self.test_ids)}")
        self.logger.info(f"Train Number: {len(self.train_ids)}, Test Number: {len(self.test_ids)}")

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        max_test_correct_num, _train_acc, _epoch = 0, 0, 0
        for epoch in range(self.params.n_epochs):
            total_loss = 0
            for batch, nf in enumerate(NeighborSampler(g=self.graph,
                                                       batch_size=self.params.batch_size,
                                                       expand_factor=self.num_neighbors,
                                                       num_hops=self.params.n_layers,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=8,
                                                       seed_nodes=self.train_ids,
                                                       transition_prob=self.graph.edata['weight'].view(
                                                           -1).cpu())):
                self.model.train()
                nf.copy_from_parent()  # Copy node/edge features from the parent graph.
                logits = self.model(nf)
                batch_nids = nf.layer_parent_nid(-1).type(torch.long).to(device=self.device)
                loss = loss_fn(logits, self.labels[batch_nids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            train_acc = self.evaluate(self.train_ids) / len(self.train_ids)
            test_correct_num = self.evaluate(self.test_ids)
            if test_correct_num > max_test_correct_num:
                max_test_correct_num = test_correct_num
                _train_acc = train_acc
                _epoch = epoch
            print(
                f">>>>Epoch {epoch:04d}: Train Acc {train_acc:.4f}, Loss {total_loss / len(self.train_ids):.4f}, Test correct {test_correct_num}, Test Acc {test_correct_num / len(self.test_ids):.4f}")

            if train_acc == 1:
                break

        print(f"---{self.params.species} {self.params.tissue} Best test result:---")
        self.logger.info(f"---{self.params.species} {self.params.tissue} Best test result:---")
        print(
            f"Epoch {_epoch:04d}, Train Acc {_train_acc:.4f}, Test Correct Num {max_test_correct_num}, Test Acc {max_test_correct_num / len(self.test_ids):.4f}")
        self.logger.info(
            f"Epoch {_epoch:04d}, Train Acc {_train_acc:.4f}, Test Correct Num {max_test_correct_num}, Test Acc {max_test_correct_num / len(self.test_ids):.4f}")

    def evaluate(self, ids):
        self.model.eval()
        total_correct = 0
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
                logits = self.model(nf)
            batch_nids = nf.layer_parent_nid(-1).type(torch.long)
            _, indices = torch.max(logits, dim=1)
            total_correct += torch.sum(indices == self.labels[batch_nids]).item()
        return total_correct


if __name__ == '__main__':
    """
    python ./code/run_atlas.py --species human --tissue Adipose --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Adrenal_gland --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Artery --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Ascending_colon --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Bladder --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Bone_marrow --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Cerebellum --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Cervix --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Chorionic_villus --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Cord_blood --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Duodenum --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Epityphlon --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Esophagus --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fallopian_tube --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Female_gonad --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_adrenal_gland --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_brain --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_calvaria --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_eye --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_heart --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_intestine --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_kidney --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_liver --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_Lung --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_male_gonad --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_muscle --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_pancreas --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_rib --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_skin --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_spinal_cord --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_stomach --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Fetal_thymus --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Gall_bladder --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Heart --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Ileum --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue JeJunum --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Kidney --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Liver --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Lung --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Muscle --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Neonatal_adrenal_gland --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Omentum --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Pancreas --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Peripheral_blood --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Placenta --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Pleura --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Prostat --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Rectum --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Sigmoid_colon --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Spleen --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Stomach --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Temporal_lobe --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Thyroid --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Trachea --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Transverse_colon --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species human --tissue Ureter --gpu "${gpu_id}" --log_file "${log_file}"
    
    
    python ./code/run_atlas.py --species mouse --tissue Bladder --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Bone_marrow --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Bone_Marrow_mesenchyme --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Brain --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Embryonic_mesenchyme --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Fetal_brain --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Fetal_intestine --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Fetal_liver --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Fetal_lung --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Fetal_stomach --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Kidney --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Liver --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Lung --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Mammary_gland --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Muscle --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Neonatal_calvaria --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Neonatal_heart --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Neonatal_muscle --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Neonatal_pancreas --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Neonatal_rib --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Neonatal_skin --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Ovary --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Pancreas --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Peripheral_blood --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Placenta --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Prostate --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Small_intestine --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Spleen --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Stomach --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Testis --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Thymus --gpu "${gpu_id}" --log_file "${log_file}"
    python ./code/run_atlas.py --species mouse --tissue Uterus --gpu "${gpu_id}" --log_file "${log_file}"
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
    parser.add_argument("--exclude_rate", type=float, default=0.005)
    # parser.add_argument("--train_dataset", nargs="+", required=True, type=int,
    #                     help="list of dataset id")
    # parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
    #                     help="list of dataset id")
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
