# DeepSort

<img src='https://img.shields.io/badge/python-3.7-brightgreen'>

### Reference-free Cell-type Annotation for Single-cell Transcriptomics using Deep Learning with a Weighted Graph Neural Network
Recent advance in single-cell RNA sequencing (scRNA-seq) has enabled large-scale transcriptional characterization of thousands of cells in multiple complex tissues, in which accurate cell type identification becomes the prerequisite and vital step for scRNA-seq studies. 

To addresses this challenge, we developed a reference-free cell-type annotation method, namely DeepSort, using a state-of-the-art deep learning algorithm, i.e. a modified graph neural network (GNN) model. It’s the first time that GNN is introduced into scRNA-seq studies and demonstrate its ground-breaking performances in this application scenario. In brief, DeepSort was constructed based on our weighted GNN framework and was then learned in two embedded high-quality scRNA-seq atlases containing 764,741 cells across 88 tissues of human and mouse, which are the most comprehensive multiple-organs scRNA-seq data resources to date.


# Dependency

<img src='https://img.shields.io/badge/scipy-1.3.1-yellowgreen'> <img src='https://img.shields.io/badge/torch-1.4.0-orange'> <img src='https://img.shields.io/badge/numpy-1.17.2-red'> <img src='https://img.shields.io/badge/pandas-0.25.1-lightgrey'> <img src='https://img.shields.io/badge/dgl-0.4.3-blue'> <img src='https://img.shields.io/badge/scikit__learn-0.22.2-green'>


- Dependencies can also be installed using `pip install -r requirements.txt`

# Install

[![download:pretrained.tar.gz](https://img.shields.io/badge/download-pretrained.tar.gz-brightgreen)](https://github.com/ZJUFanLab/DeepSort/releases/download/v2.0/pretrained.tar.gz)

1. Download source codes of DeepSort.
2. Download pretrained models from the release page and uncompress them.
```
tar -xzvf pretrained.tar.gz
```

After executing the above steps, the final DeepSort tree should look like this:
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

To test one data file `human_Pancreas11.csv`, you should execute the following command:
```shell script
python run.py --species human --tissue Pancreas --test_dataset 11 --gpu -1 --threshold 0
```
- ``--species`` The species of cells, `human` or `mouse`.
- ``--tissue`` The tissue of cells. see __Details__
- ``--test_dataset`` The dataset to be tested, in other words, as the file naming rule states, it is exactly the number of cells in the data file.
- ``--gpu`` Specify the GPU to use, `-1` for cpu.
- ``--threshold`` The threshold that constitutes the edge in the graph, default is 0.

### Output
For each test dataset, it will output a `.csv` file named as `species_Tissue_Number.csv` under the `result` directory. For example, output of test dataset `human_Pancreas11_data.csv` is `human_Pancreas_11.csv`

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
python run.py --species human --tissue Pancreas --test_dataset 11 --gpu -1 --threshold 0
```

```
python run.py --species mouse --tissue Intestine --test_dataset 28 --gpu -1 --threshold 0
```


