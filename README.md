# DeepSort

### Reference-free Cell-type Annotation for Single-cell Transcriptomics using Deep Learning with a Weighted Graph Neural Network
Recent advance in single-cell RNA sequencing (scRNA-seq) has enabled large-scale transcriptional characterization of thousands of cells in multiple complex tissues, in which accurate cell type identification becomes the prerequisite and vital step for scRNA-seq studies. Here, we introduce DeepSort, a reference-free cell-type annotation tool for single-cell transcriptomics that uses a deep learning model with a weighted graph neural network.

## Install
### Dependencies
- Compatible with Python 3.7 and Pytorch 1.4
- Dependencies can be installed using `pip install -r requirements.txt`

1. Execute `fdfjsdklf` for unpacking the pre-trained models.
2.
3. After unpacking, your tree should look like this:
```
 |- pretrianed
     |- human
        |- graphs
        |- statistics
        |- map.xlsx
        |- models
     |- mouse
        |- graphs
        |- statistics
        |- map.xlsx
        |- models
 |- models
 |- utils
 |- run.py
 |- requirements.txt
 |- README.md
```

## Usage

### Prepare test data

1. The file name of test data should be named in this format: **species_TissueNumber_data.csv**. For example, `human_Spleen9887_data.csv` is a data file contains 9887 human spleen cells.
2. ***How to describe the format of data??***
3. All the test data should be included in a species specific dictionary under the `pretrianed` dictionary . If you name the dictionary of human data `test`, and then your file tree should look like this:
```
 |- pretrianed
     |- human
        |- graphs
        |- statistics
        |- map.xlsx
        |- models
        |- test
            |- ....
 ....
```

### Run
```shell script
python run.py --species human --tissue Lung --test_dataset 2064 6338 7211 10743 --gpu 0 --log_dir logs --log_file log --test_dir test --save_dir result
```
- ``--species`` The species of cells, `human` or `mouse`.
- ``--tissue`` The tissue of cells.
- ``--test_dataset`` The dataset to be tested, in other words, as the file naming rule states, it is exactly the number of cells in the data file.
- ``--gpu`` Specify the GPU to use, `-1` for cpu.
- ``--log_dir`` Dictionary of log files.
- ``--log_file`` Name of log file.
- ``--test_dir`` The dictionary name of the test data. 
- ``--save_dir`` Dictionary in which the annotation result outputs.

### Output
