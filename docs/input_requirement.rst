Input Requirement
=================

Data File
---------

scDeepSort assumes data to be in the form of 2D table of shape (``num_of_genes``, ``num_of_cells``) with index and header included. For example:

.. list-table:: Input Data Format
   :widths: 25 25 25 25 25
   :header-rows: 1

   * -
     - Cell 1
     - Cell 2
     - ...
     - Cell N
   * - Gene 1
     - 0
     - 2.4
     - ...
     - 5.0
   * - Gene 2
     - 0.8
     - 1.1
     - ...
     - 4.3
   * - Gene 3
     - ...
     - ...
     - ...
     - ...
   * - Gene M
     - 1.8
     - 0
     - ...
     - 0

We recommend csv format for input data matrix. The input data matrix should be pre-processed by first revising gene symbols according to `NCBI Gene Database <https://www.ncbi.nlm.nih.gov/gene>`_ updated on Jan. 10, 2020, wherein unmatched genes and duplicated genes will be removed. Then it should be normalized with the default `LogNormalize` method in `Seurat` (R package), detailed in `R Script <https://github.com/ZJUFanLab/scDeepSort/blob/dev/pre-process.R>`_, wherein the column represents each cell and the row represent each gene for training and testing data, as shown above.

scDeepSort will conduct internal check and transformation on the input data according to given data paths.

Cell Type File
--------------
We assume csv format for all cell type files.


.. code-block:: python

    "","Cell","Cell_type"
    "1","Cell 1","Conventional CD4+ T cell"
    "2","Cell 2","Conventional CD4+ T cell"
    ...
    "N","Cell N","Conventional CD4+ T cell"


