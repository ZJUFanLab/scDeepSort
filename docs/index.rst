scDeepSort Documentation
------------------------

**scDeepSort: A Pre-trained Cell-type Annotation Method for Single-cell Transcriptomics using Deep Learning with a Weighted Graph Neural Network**

Recent advance in single-cell RNA sequencing (scRNA-seq) has enabled large-scale transcriptional characterization of thousands of cells in multiple complex tissues, in which accurate cell type identification becomes the prerequisite and vital step for scRNA-seq studies. 

To addresses this challenge, we developed a pre-trained cell-type annotation method, namely scDeepSort, using a state-of-the-art deep learning algorithm, i.e. a modified graph neural network (GNN) model. It’s the first time that GNN is introduced into scRNA-seq studies and demonstrate its ground-breaking performances in this application scenario. In brief, scDeepSort was constructed based on our weighted GNN framework and was then learned in two embedded high-quality scRNA-seq atlases containing 764,741 cells across 88 tissues of human and mouse, which are the most comprehensive multiple-organs scRNA-seq data resources to date. For more information, please refer to a preprint in [bioRxiv 2020.05.13.094953.](https://www.biorxiv.org/content/10.1101/2020.05.13.094953v1)

Reference
---------

.. code-block:: latex

    @article{shao2020reference,
      title={Reference-free Cell-type Annotation for Single-cell Transcriptomics using Deep Learning with a Weighted Graph Neural Network},
      author={Shao, Xin and Yang, Haihong and Zhuang, Xiang and Liao, Jie and Yang, Yueren and Yang, Penghui and Cheng, Junyun and Lu, Xiaoyan and Chen, Huajun and Fan, Xiaohui},
      journal={bioRxiv},
      year={2020},
      publisher={Cold Spring Harbor Laboratory}
    }

.. toctree::
   :maxdepth: 1
   :caption: For Users

   How to Get Started <get_started>
   Installation Guide <installation>
   Input Requirement <input_requirement>
   API Reference <api>

.. toctree::
   :maxdepth: 1
   :caption: About

   Contact <contact>

---------------
