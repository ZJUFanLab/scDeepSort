API Reference
=============

Below is the class and function reference for scDeepSort. Notice that the package is under active development, and some features may not be stable yet.

DeepSortPredictor
-----------------

.. code-block:: python

    # Class Definition
    DeepSortPredictor(species,
                      tissue,
                      file_type='csv',
                      unsure_rate=2.)

- **species**: The species of cells, ``human`` or ``mouse``.
- **tissue**: The tissue of cells. For the detailed list of supported tissue of our model, please refer to `GitHub Wiki page <https://github.com/ZJUFanLab/scDeepSort/wiki>`_.
- **file_type**: The format of data file, ``csv`` or ``gz``. csv for .csv files and gz for .gz files. For details, please refer to `R script <https://github.com/ZJUFanLab/scDeepSort/blob/master/pre-process.R>`_.
- **unsure_rate**: The multiplier for unsure threshold (computed as unsure_rate / num_classes) to cast the type of cell to the unsure type, default to ``2.0``. Set it as 0 to exclude the unsure type.

.. code-block:: python

    # Class Method
    DeepSortPredictor.predict(input_file, 
                              save_path=None) -> pandas.DataFrame

- **input_file**: The file path for test dataset.
- **save_path**: The destination for saving predictions. Save results to disk if path provided.

Example
*******
.. code-block:: python

    from deepsort import DeepSortPredictor
    # define the model
    model = DeepSortPredictor(species='human',
                              tissue='Brain')
    # use the trained model to predict
    test_files = ['/path/to/human_brain_test_data_1.csv', '/path/to/human_brain_test_data_2.csv']
    for test_file in test_files:
        model.predict(test_file, save_path='results', model_path='model_save_path')
        
DeepSortClassifier
------------------

.. code-block:: python

    # Class Definition
    DeepSortClassifier(species,
                       tissue,
                       dense_dim=400,
                       hidden_dim=200,
                       batch_size=256,
                       dropout=0.1,
                       gpu_id=-1,
                       file_type='csv',
                       learning_rate=0.001,
                       weight_decay=5e-4,
                       n_epochs=300,
                       n_layers=1,
                       threshold=0,
                       num_neighbors=None,
                       exclude_rate=0.005,
                       random_seed=None,
                       validation_fraction=0.1)

- **species**: The species of cells, ``human`` or ``mouse``.
- **tissue**: The tissue of cells. For the detailed list of supported tissue of our model, please refer to `GitHub Wiki page <https://github.com/ZJUFanLab/scDeepSort/wiki>`_.
- **dense_dim**: The initial dimension of node embedding for cells and genes. Default to ``400``.
- **hidden_dim**: The hidden dimension of Weighted Graph Aggregator Layer. Default to ``200``.
- **batch_size**: The number of samples per batch. Default to ``256``.
- **dropout**: The dropout rate for the output representation of Weighted Graph Aggregator Layer, default to ``0.1``.
- **gpu_id**: The GPU id for training and testing. -1 for CPU. Default to ``-1``.
- **file_type**: The format of data file, ``csv`` or ``gz``. csv for .csv files and gz for .gz files.
- **learning_rate**: The learning rate of optimizer. Default to ``0.001``.
- **weight_decay**: The weight decay of optimizer. Default to ``0.00004``.
- **n_epochs**: Maximum number of epochs. Default to ``300``. 
- **n_layers**: The number of layers. Default to ``1``.
- **threshold**: The weight threshold for edges between cell nodes and gene nodes. Default to ``0``.
- **num_neighbors**: The number of sampled neighbors per nodes in training time. If ``None``, all the neighbors will be sampled.
- **exclude_rate**: Exclude a class if the portion of this class is less than ``exclude_rate``. Default to ``0.005``.
- **random_seed**: For reproducibility. Fixed if given. Default to ``None``.
- **validation_fraction**: The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Default to ``0.1``.

.. code-block:: python

    # Class Method
    DeepSortClassifier.fit(files, 
                           save_path=None)

- **files**: The file path for training datasets. We assume ``files`` in the form of ``list of (data_file, celltype_file)``.
- **save_path**: The destination for saving models.

.. code-block:: python

    # Class Method
    DeepSortClassifier.predict(input_file, 
                               model_path, 
                               save_path=None,
                               unsure_rate=2., 
                               file_type='csv')  -> pandas.DataFrame

- **input_file**: The file path for test dataset.
- **model_path**: The path for loading saved models.
- **save_path**: The destination for saving predictions. Save results to disk if path provided.

Example
*******

.. code-block:: python

    from deepsort import DeepSortClassifier
    # define the model
    model = DeepSortClassifier(species='human',
                               tissue='Brain',
                               dense_dim=50,
                               hidden_dim=20,
                               gpu_id=0,
                               n_layers=2,
                               random_seed=1,
                               n_epochs=20)
    train_files = [('/path/to/human_brain_data_1.csv', '/path/to/human_brain_celltype_1.csv'),
                   ('/path/to/human_brain_data_2.csv', '/path/to/human_brain_celltype_2.csv')]
    test_files = ['/path/to/human_brain_test_data_1.csv', '/path/to/human_brain_test_data_2.csv']
    # fit the model
    model.fit(train_files, save_path='model_save_path')
    # use the saved model to predict
    for test_file in test_files:
        model.predict(test_file, save_path='results', model_path='model_save_path')

