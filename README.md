# scDeepSort

[![python 3.7](https://img.shields.io/badge/python-3.7-brightgreen)](https://www.python.org/)

### Reference-free Cell-type Annotation for Single-cell Transcriptomics using Deep Learning with a Weighted Graph Neural Network
Recent advance in single-cell RNA sequencing (scRNA-seq) has enabled large-scale transcriptional characterization of thousands of cells in multiple complex tissues, in which accurate cell type identification becomes the prerequisite and vital step for scRNA-seq studies. 

To addresses this challenge, we developed a reference-free cell-type annotation method, namely scDeepSort, using a state-of-the-art deep learning algorithm, i.e. a modified graph neural network (GNN) model. It’s the first time that GNN is introduced into scRNA-seq studies and demonstrate its ground-breaking performances in this application scenario. In brief, scDeepSort was constructed based on our weighted GNN framework and was then learned in two embedded high-quality scRNA-seq atlases containing 764,741 cells across 88 tissues of human and mouse, which are the most comprehensive multiple-organs scRNA-seq data resources to date. For more information, please refer to a preprint in [bioRxiv 2020.05.13.094953.](https://www.biorxiv.org/content/10.1101/2020.05.13.094953v1)

# Install

[![download:pretrained.tar.gz](https://img.shields.io/badge/download-pretrained.tar.gz-blue)](https://github.com/ZJUFanLab/scDeepSort/releases/download/v2.0/pretrained.tar.gz)

1. Download source codes of scDeepSort.
2. Download pretrained models from the release page and uncompress them.
```
tar -xzvf pretrained.tar.gz
```

After executing the above steps, the final scDeepSort tree should look like this:
```
 |- pretrianed
     |- human
        |- graphs
        |- statistics
        |- models
     |- mouse
        |- graphs
        |- statistics
        |- models
 |- test
     |- human
     |- mouse
 |- map
    |- human
        |- map.xlsx
    |- mouse
        |- map.xlsx
 |- models
    |- __init__.py
    |- gnn.py
 |- utils
    |- __init__.py
    |- preprocess.py
 |- run.py
 |- requirements.txt
 |- README.md
```

# Dependency
[![scipy-1.3.1](https://img.shields.io/badge/scipy-1.3.1-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.4.0](https://img.shields.io/badge/torch-1.4.0-orange)](https://github.com/pytorch/pytorch) [![numpy-1.17.2](https://img.shields.io/badge/numpy-1.17.2-red)](https://github.com/numpy/numpy) [![pandas-0.25.1](https://img.shields.io/badge/pandas-0.25.1-lightgrey)](https://github.com/pandas-dev/pandas) [![dgl-0.4.3](https://img.shields.io/badge/dgl-0.4.3-blue)](https://github.com/dmlc/dgl) [![scikit__learn-0.22.2](https://img.shields.io/badge/scikit__learn-0.22.2-green)](https://github.com/scikit-learn/scikit-learn)

- Dependencies can also be installed using `pip install -r requirements.txt`

# Usage

### Prepare test data

1. The file name of test data should be named in this format: **species_TissueNumber_data.csv**. For example, `human_Pancreas11_data.csv` is a data file containing 11 human pancreas cells.
2. The test single-cell transcriptomics csv data file should be normalized with the defalut `LogNormalize` method with `Seurat` (R package), wherein the column represents each cell and the row represent each gene, as shown below.

      |          |Cell 1|Cell 2|Cell 3|...  |
      | :---:    |:---: | :---:| :---:|:---:|
      |__Gene 1__|    0 | 2.4  |  5.0 |...  |
      |__Gene 2__| 0.8  | 1.1  |  4.3 |...  |
      |__Gene 3__|1.8   |    0 |  0   |...  |
      |  ...     |  ... |  ... | ...  |...  |



3. All the test data should be included under the `test` directory. Furthermore, all of the human testing datasets and mouse testing datasets are required to be under `./test/human` and `./test/mouse` respectively.

### Run

#### Evaluate
To evaluate one data file `mouse_Liver4122_data.csv`, which we report the prediction accuracy in the paper, you should execute the following command:
```shell script
python run.py --species human --tissue Liver --test_dataset 4122 --gpu -1 --evaluate
```
- ``--species`` The species of cells, `human` or `mouse`.
- ``--tissue`` The tissue of cells. see __Details__
- ``--test_dataset`` The dataset to be tested, in other words, as the file naming rule states, it is exactly the number of cells in the data file.
- ``--gpu`` Specify the GPU to use, `-1` for cpu.

#### Test
To test one data file `human_Pancreas11.csv`, you should execute the following command:
```shell script
python run.py --species human --tissue Pancreas --test_dataset 11 --gpu -1 --test
```
- ``--species`` The species of cells, `human` or `mouse`.
- ``--tissue`` The tissue of cells. see __Details__
- ``--test_dataset`` The dataset to be tested, in other words, as the file naming rule states, it is exactly the number of cells in the data file.
- ``--gpu`` Specify the GPU to use, `-1` for cpu.

### Output
For each dataset, it will output a `.csv` file named as `species_Tissue_Number.csv` under the `result` directory. For example, output of test dataset `human_Pancreas11_data.csv` is `human_Pancreas_11.csv`

**Evaluate:**

Each line of the output file corresponds to the original cell type and predictive cell type.

**Test:**

Each line of the output file corresponds to the predictive cell type.

### Details

<img src='https://img.shields.io/badge/human-tissue-red.svg'>

- Adipose
- Adrenal_gland
- Artery
- Ascending_colon
- Bladder
- Blood
- Bone_marrow
- Brain
- Cervix
- Chorionic_villus
- Colorectum
- Cord_blood
- Epityphlon
- Esophagus
- Fallopian_tube
- Female_gonad
- Fetal_adrenal_gland
- Fetal_brain
- Fetal_calvaria
- Fetal_eye
- Fetal_heart
- Fetal_intestine
- Fetal_kidney
- Fetal_liver
- Fetal_Lung
- Fetal_male_gonad
- Fetal_muscle
- Fetal_pancreas
- Female_gonad
- Fetal_rib
- Fetal_skin
- Fetal_spinal_cord
- Fetal_stomach
- Fetal_thymus
- Gall_bladder
- Heart
- Kidney
- Liver
- Lung
- Muscle
- Neonatal_adrenal_gland
- Omentum
- Pancreas
- Placenta
- Pleura
- Prostat
- Spleen
- Stomach
- Temporal_lobe
- Thyroid
- Trachea
- Ureter

<img src='https://img.shields.io/badge/mouse-tissue-red.svg'>

- Bladder
- Blood
- Bone_marrow
- Bone_Marrow_mesenchyme
- Brain
- Embryonic_mesenchyme
- Fetal_brain
- Fetal_intestine
- Fetal_liver
- Fetal_lung
- Fetal_stomach
- Intestine
- Kidney
- Liver
- Lung
- Mammary_gland
- Muscle
- Neonatal_calvaria
- Neonatal_heart
- Neonatal_muscle
- Neonatal_pancreas
- Neonatal_rib
- Neonatal_skin
- Ovary
- Pancreas
- Placenta
- Prostate
- Spleen
- Stomach
- Testis
- Thymus
- Uterus

# Examples
```
python run.py --species human --tissue Pancreas --test_dataset 11 --gpu -1 --test
```

```
python run.py --species mouse --tissue Intestine --test_dataset 28 --gpu -1 --test
```

To test all datasets, please download [human_test_data](https://github.com/ZJUFanLab/DeepSort/releases/download/v2.0/human_test_data.7z) and [mouse_test_data](https://github.com/ZJUFanLab/DeepSort/releases/download/v2.0/mouse_test_data.7z) in [release page](https://github.com/ZJUFanLab/DeepSort/releases)
