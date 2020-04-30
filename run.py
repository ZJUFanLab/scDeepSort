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
from utils import load_data, get_logger
from models import GNN
from pprint import pprint


class Trainer:
    def __init__(self, params):
        self.params = params
        self.postfix = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        self.prj_path = Path(__file__).parent.resolve()
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.logger = get_logger(self.prj_path / self.params.log_dir, self.params.log_file)
        self.total_cell, self.num_genes, self.num_classes, self.map_dict, self.id2label, self.test_dict = load_data(
            params)
        """
        test_dict = {
            'graph': test_graph_dict,
            'label': test_label_dict,
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

        self.logger.info("========================================")
        self.logger.info(vars(self.params))

    def run(self):
        record = dict()
        for num in self.params.test_dataset:
            c, t, test_acc, pred = self.evaluate_test(num)
            record[num] = dict()
            record[num]['c'], record[num]['t'], record[num]['acc'], record[num]['pred'] = c, t, test_acc, pred
        for num in self.params.test_dataset:
            print(
                f"#{num} Test Acc: {record[num]['acc']:.4f}, [{record[num]['c']}/{record[num]['t']}]")
            self.logger.info(
                f"#{num} Test Acc: {record[num]['acc']:.4f}, [{record[num]['c']}/{record[num]['t']}]")
        self.save_pred(record)

    def load_model(self):
        model_path = self.prj_path / 'checkpoint' / self.params.model_name
        state = torch.load(model_path)
        self.model.load_state_dict(state['model'])
        # self.optimizer.load_state_dict(state['optimizer'])

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
        indices = new_logits.numpy().argmax(axis=1)
        predict_label = self.id2label[indices]
        correct = 0
        for p_label, t_label in zip(predict_label, self.test_dict['label'][num]):
            if p_label in self.map_dict[num][t_label]:
                correct += 1
        total = predict_label.shape[0]
        return correct, total, correct / total, predict_label

    def save_pred(self, record):
        save_path = self.prj_path / self.params.save_dir
        if not save_path.exists():
            save_path.mkdir()
        for num in self.params.test_dataset:
            df = pd.DataFrame({'original label': self.test_dict['label'][num],
                               'prediction': record[num]['pred']})
            df.to_csv(
                save_path / (self.params.species + f"_{self.params.tissue}_{num}_" + self.postfix + ".csv"),
                index=False)


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
     
    python run.py --model_name mouse-Pancreas-9-30_04_2020_18:07:36.pt --test_dataset 1354 108 131 207 --tissue Pancreas

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
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--dense_dim", type=int, default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="number of hidden gcn units")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--test_dataset", nargs="+", required=True, type=int,
                        help="list of dataset id")
    parser.add_argument("--species", default='mouse', type=str)
    parser.add_argument("--tissue", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--log_file", type=str, default='out')
    parser.add_argument("--test_dir", type=str, default='test')
    parser.add_argument("--save_dir", type=str, default='result')
    parser.add_argument("--model_name", type=str, required=True)
    params = parser.parse_args()
    pprint(vars(params))

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    trainer = Trainer(params)
    trainer.run()
