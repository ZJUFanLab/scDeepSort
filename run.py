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


class Runner:
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
        # model_path = self.prj_path / 'checkpoint' / self.params.model_name
        model_path = self.prj_path / 'pretrained' / self.params.species / 'models' / self.params.model_name
        state = torch.load(model_path, map_location=self.device)
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
     
    python run.py --model_name human-Blood-163-30_04_2020_22_58_36.pt --test_dataset 9649 3223 2469 --tissue Blood
    python run.py --model_name human-Brain-12-30_04_2020_23_17_55.pt --test_dataset 251 2892 1834 --tissue Brain
    python run.py --model_name human-Colorectum-18-01_05_2020_00_44_09.pt --test_dataset 94 11894 --tissue Colorectum
    python run.py --model_name human-Lung-146-01_05_2020_05_33_15.pt --test_dataset 2064 6338 7211 10743 11204 4624 2841 9566 --tissue Lung
    python run.py --model_name human-Pancreas-279-01_05_2020_05_53_29.pt --test_dataset 465 958 20 185 15 11 --tissue Pancreas
    python run.py --model_name human-Spleen-289-01_05_2020_08_22_02.pt --test_dataset 11081 9887 18513 16286 14848 --tissue Spleen
    
    python run.py --model_name mouse-Blood-56-30_04_2020_22_24_33.pt --test_dataset 768 1109 1223 1610 --tissue Blood --log_file "${log_file}"
    python run.py --model_name mouse-Bone_marrow-274-30_04_2020_23_54_44.pt --test_dataset 47 467 --tissue Bone_marrow --log_file "${log_file}"
    python run.py --model_name mouse-Brain-82-01_05_2020_00_35_39.pt --test_dataset 19431 2695 2502 2545 3005 4397 --tissue Brain --log_file "${log_file}"
    python run.py --model_name mouse-Fetal_brain-179-01_05_2020_00_48_08.pt --test_dataset 369 --tissue Fetal_brain --log_file "${log_file}"
    python run.py --model_name mouse-Intestine-60-01_05_2020_01_07_25.pt --test_dataset 28 192 260 3260 1449 529 --tissue Intestine --log_file "${log_file}"
    python run.py --model_name mouse-Kidney-53-01_05_2020_01_24_23.pt --test_dataset 203 7926 1435 --tissue Kidney --log_file "${log_file}"
    python run.py --model_name mouse-Liver-39-01_05_2020_01_51_02.pt --test_dataset 3729 4122 7761 18000 --tissue Liver --log_file "${log_file}"
    python run.py --model_name mouse-Lung-28-01_05_2020_02_33_16.pt --test_dataset 769 707 1920 6340 --tissue Lung --log_file "${log_file}"
    python run.py --model_name mouse-Mammary_gland-35-01_05_2020_02_52_22.pt --test_dataset 133 --tissue Mammary_gland --log_file "${log_file}"
    python run.py --model_name mouse-Pancreas-62-01_05_2020_03_29_05.pt --test_dataset 1354 108 131 207 --tissue Pancreas --log_file "${log_file}"
    python run.py --model_name mouse-Spleen-73-01_05_2020_03_35_50.pt --test_dataset 1759 1433 1081 --tissue Spleen --log_file "${log_file}"
    python run.py --model_name mouse-Testis-45-01_05_2020_12_33_53.pt --test_dataset 2584 4239 8792 9923 6598 300 4233 1662 299 199 398 296 --tissue Testis --log_file "${log_file}"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    # parser.add_argument("--dropout", type=float, default=0.1,
    #                     help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU id, -1 for cpu")
    # parser.add_argument("--dense_dim", type=int, default=400,
    #                     help="number of hidden gcn units")
    # parser.add_argument("--hidden_dim", type=int, default=200,
    #                     help="number of hidden gcn units")
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
    params.dropout = 0.1
    params.dense_dim = 400
    params.hidden_dim = 200
    pprint(vars(params))

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    trainer = Runner(params)
    trainer.run()
