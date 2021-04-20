# scDeepSort

[![python 3.7](https://img.shields.io/badge/python-3.7-brightgreen)](https://www.python.org/) [![R >3.6](https://img.shields.io/badge/R-%3E3.6-blue)](https://www.r-project.org/) 

### Cell-type Annotation for Single-cell Transcriptomics using Deep Learning with a Weighted Graph Neural Network
Recent advance in single-cell RNA sequencing (scRNA-seq) has enabled large-scale transcriptional characterization of thousands of cells in multiple complex tissues, in which accurate cell type identification becomes the prerequisite and vital step for scRNA-seq studies. 

To addresses this challenge, we developed a pre-trained cell-type annotation method, namely scDeepSort, using a state-of-the-art deep learning algorithm, i.e. a modified graph neural network (GNN) model. It’s the first time that GNN is introduced into scRNA-seq studies and demonstrate its ground-breaking performances in this application scenario. In brief, scDeepSort was constructed based on our weighted GNN framework and was then learned in two embedded high-quality scRNA-seq atlases containing 764,741 cells across 88 tissues of human and mouse, which are the most comprehensive multiple-organs scRNA-seq data resources to date. For more information, please refer to a preprint in [bioRxiv 2020.05.13.094953.](https://www.biorxiv.org/content/10.1101/2020.05.13.094953v1)

# Install

[![scipy-1.3.1](https://img.shields.io/badge/scipy-1.3.1-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.4.0](https://img.shields.io/badge/torch-1.4.0-orange)](https://github.com/pytorch/pytorch) [![numpy-1.17.2](https://img.shields.io/badge/numpy-1.17.2-red)](https://github.com/numpy/numpy) [![pandas-0.25.1](https://img.shields.io/badge/pandas-0.25.1-lightgrey)](https://github.com/pandas-dev/pandas) [![dgl-0.4.3](https://img.shields.io/badge/dgl-0.4.3-blue)](https://github.com/dmlc/dgl) [![scikit__learn-0.22.2](https://img.shields.io/badge/scikit__learn-0.22.2-green)](https://github.com/scikit-learn/scikit-learn) [![xlrd-1.2.0](https://img.shields.io/badge/xlrd-1.2.0-yellow)](https://github.com/python-excel/xlrd)

Download [`scDeepSort-v1.0-cu102.tar.gz`]() from the [release](https://github.com/ZJUFanLab/scDeepSort/releases) page and execute the following command:
```
pip install scDeepSort-v1.0-cu102.tar.gz
```

# Usage

The test single-cell transcriptomics csv data file should be pre-processed by first revising gene symbols according to [NCBI Gene database](https://www.ncbi.nlm.nih.gov/gene) updated on Jan. 10, 2020, wherein unmatched genes and duplicated genes will be removed. Then the data should be normalized with the defalut `LogNormalize` method in `Seurat` (R package), detailed in [`pre-process.R`](https://github.com/ZJUFanLab/scDeepSort/blob/dev/pre-process.R), wherein the column represents each cell and the row represent each gene for final test data, as shown below. 


      |          |Cell 1|Cell 2|Cell 3|...  |
      | :---:    |:---: | :---:| :---:|:---:|
      |__Gene 1__|    0 | 2.4  |  5.0 |...  |
      |__Gene 2__| 0.8  | 1.1  |  4.3 |...  |
      |__Gene 3__|1.8   |    0 |  0   |...  |
      |  ...     |  ... |  ... | ...  |...  |


## Predict using pre-trained models

1. The file name of test data should be named in this format: **species_TissueNumber_data.csv**. For example, `human_Pancreas11_data.csv` is a data file containing 11 human pancreas cells.
2. The test single-cell transcriptomics csv data file should be pre-processed by first revising gene symbols according to [NCBI Gene database](https://www.ncbi.nlm.nih.gov/gene) updated on Jan. 10, 2020, wherein unmatched genes and duplicated genes will be removed. Then the data should be normalized with the defalut `LogNormalize` method in `Seurat` (R package), detailed in [`pre-process.R`](https://github.com/ZJUFanLab/scDeepSort/blob/dev/pre-process.R), wherein the column represents each cell and the row represent each gene for final test data, as shown below. 

      |          |Cell 1|Cell 2|Cell 3|...  |
      | :---:    |:---: | :---:| :---:|:---:|
      |__Gene 1__|    0 | 2.4  |  5.0 |...  |
      |__Gene 2__| 0.8  | 1.1  |  4.3 |...  |
      |__Gene 3__|1.8   |    0 |  0   |...  |
      |  ...     |  ... |  ... | ...  |...  |

3. All the test data should be included under the `test` directory. Human datasets should be under `./test/human` and mouse datasets should be under `./test/mouse`

### Evaluate
Use `--evaluate` to reproduce the results as shown in our paper. For example,
to evaluate the data `mouse_Testis199_data.csv`, you should execute the following command:

```
python predict.py --species human --tissue Testis --test_dataset 199 --gpu -1 --evaluate --filetype gz --unsure_rate 2
```

- `--species` The species of cells, `human` or `mouse`.

- `--tissue` The tissue of cells. See [wiki page](https://github.com/ZJUFanLab/scDeepSort/wiki)

- `--test_dataset` The number of cells in the test data.

- `--gpu` Specify the GPU to use, `0` for gpu,`-1` for cpu.

- `--filetype` The format of datafile, `csv` for `.csv` files and `gz` for `.gz` files. See [`pre-process.R`](https://github.com/ZJUFanLab/scDeepSort/blob/dev/pre-process.R)

- `--unsure_rate` The threshold to define the unsure type, default is 2. Set it as 0 to exclude the unsure type.

__Output:__ the output named as `species_Tissue_Number.csv` will be under the automatically generated `result` directory, which contains four columns, the first is the cell id, the second is the original cell type, the third is the predicted main type, the fourth is the predicted subtype if applicable.

__Note:__ to evaluate all testing datasets in our paper, please download them in [release page](https://github.com/ZJUFanLab/DeepSort/releases)

### Test
Use `--test` to test your own datasets. For example,
to test the data `human_Pancreas11_data.csv`, you should execute the following command:

```
python predict.py --species human --tissue Pancreas --test_dataset 11 --gpu -1 --test --filetype csv --unsure_rate 2
```
- `--species` The species of cells, `human` or `mouse`.

- `--tissue` The tissue of cells. See [wiki page](https://github.com/ZJUFanLab/scDeepSort/wiki)

- `--test_dataset` The number of cells in the test data.

- `--gpu` Specify the GPU to use, `0` for gpu, `-1` for cpu.

- `--filetype` The format of datafile, `csv` for `.csv` files and `gz` for `.gz` files. See [`pre-process.R`](https://github.com/ZJUFanLab/scDeepSort/blob/dev/pre-process.R)

- `--unsure_rate` The threshold to define the unsure type, default is 2. Set it as 0 to exclude the unsure type.

__Output:__ the output named as `species_Tissue_Number.csv` will be under the automatically generated `result` directory, which contains three columns, the first is the cell id, the second is the predicted main type, the third is the predicted subtype if applicable.

## Train your own model and predict
To train your own model, you should prepare two files, i.e., a data file as descrived above, and a cell annotation file under the `./train` directory as the example files. Then execute the following command:

```
python train.py --species human --tissue Adipose --gpu -1 --filetype gz
```

```
python train.py --species mouse --tissue Muscle --gpu -1 --filetype gz
```

- `--species` The species of cells, `human` or `mouse`.

- `--tissue` The tissue of cells.

- `--gpu` Specify the GPU to use, `0` for gpu, `-1` for cpu.

- `--filetype` The format of datafile, `csv` for `.csv` files and `gz` for `.gz` files. See `pre-process.R`

__Output:__ the trained model will be under the `pretrained` directory, which can be used to test new datasets on the same tissue using `predict.py` as described above. 

# About

scDeepSort manuscript is under major revision. For more information, please refer to the preprint in [bioRxiv 2020.05.13.094953.](https://www.biorxiv.org/content/10.1101/2020.05.13.094953v1). Should you have any questions, please contact Xin Shao at xin_shao@zju.edu.cn, Haihong Yang at capriceyhh@zju.edu.cn, or Xiang Zhuang at 3160105000@zju.edu.cn