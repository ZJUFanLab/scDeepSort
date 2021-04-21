# scDeepSort

[![python 3.7](https://img.shields.io/badge/python-3.7-brightgreen)](https://www.python.org/) [![R >3.6](https://img.shields.io/badge/R-%3E3.6-blue)](https://www.r-project.org/) 
Document detailed in https://scdeepsort.readthedocs.io/en/master/index.html
### Cell-type Annotation for Single-cell Transcriptomics using Deep Learning with a Weighted Graph Neural Network
Recent advance in single-cell RNA sequencing (scRNA-seq) has enabled large-scale transcriptional characterization of thousands of cells in multiple complex tissues, in which accurate cell type identification becomes the prerequisite and vital step for scRNA-seq studies. 

To addresses this challenge, we developed a pre-trained cell-type annotation method, namely scDeepSort, using a state-of-the-art deep learning algorithm, i.e. a modified graph neural network (GNN) model. In brief, scDeepSort was constructed based on our weighted GNN framework and was then learned in two embedded high-quality scRNA-seq atlases containing 764,741 cells across 88 tissues of human and mouse, which are the most comprehensive multiple-organs scRNA-seq data resources to date. For more information, please refer to a preprint in [bioRxiv 2020.05.13.094953.](https://www.biorxiv.org/content/10.1101/2020.05.13.094953v1)

# Install

[![scipy-1.3.1](https://img.shields.io/badge/scipy-1.3.1-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.6.0](https://img.shields.io/badge/torch-1.6.0-orange)](https://github.com/pytorch/pytorch) [![numpy-1.17.2](https://img.shields.io/badge/numpy-1.17.2-red)](https://github.com/numpy/numpy) [![pandas-0.25.1](https://img.shields.io/badge/pandas-0.25.1-lightgrey)](https://github.com/pandas-dev/pandas) [![dgl-0.4.3](https://img.shields.io/badge/dgl-0.4.3-blue)](https://github.com/dmlc/dgl) [![scikit__learn-0.22.2](https://img.shields.io/badge/scikit__learn-0.22.2-green)](https://github.com/scikit-learn/scikit-learn) [![xlrd-1.2.0](https://img.shields.io/badge/xlrd-1.2.0-yellow)](https://github.com/python-excel/xlrd)

We provide CPU and CUDA builds, If you want to install scDeepSort with a CPU build, please download [`scDeepSort-v1.0-cpu.tar.gz`](https://github.com/ZJUFanLab/scDeepSort/releases) from the [release](https://github.com/ZJUFanLab/scDeepSort/releases) page and execute the following command:
```
pip install scDeepSort-v1.0-cpu.tar.gz
```
For more details, see [installation guide](https://scdeepsort.readthedocs.io/en/master/installation.html) in the document.
# Usage

The test single-cell transcriptomics csv data file should be pre-processed by first revising gene symbols according to [NCBI Gene database](https://www.ncbi.nlm.nih.gov/gene) updated on Jan. 10, 2020, wherein unmatched genes and duplicated genes will be removed. Then the data should be normalized with the defalut `LogNormalize` method in `Seurat` (R package), detailed in [`pre-process.R`](https://github.com/ZJUFanLab/scDeepSort/blob/dev/pre-process.R).

- Predict using pre-trained models [`DeepSortPredictor`](https://scdeepsort.readthedocs.io/en/master/api.html#deepsortpredictor)

- Train your own model and predict [`DeepSortClassifier`](https://scdeepsort.readthedocs.io/en/master/api.html#deepsortclassifier)

Please refer to the [document](https://scdeepsort.readthedocs.io/en/master/index.html) of scDeepSort for detailed guidence using scDeepSort as a python package. 

# About
scDeepSort manuscript is under major revision. Should you have any questions, please contact Xin Shao at xin_shao@zju.edu.cn, Haihong Yang at capriceyhh@zju.edu.cn, or Xiang Zhuang at zhuangxiang@zju.edu.cn