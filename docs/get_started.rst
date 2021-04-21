Getting Started
==================

This is a quick start guide for you to try out scDeepSort. The full script is available at :ref:`script`.


1. Evaluate with Pre-trained Models
***********************************

Define the Model
----------------

scDeepSort provides unified APIs on evaluating different datasets with pre-trained models.

For the demo on cell type annotating, the corresponding model is ``DeepSortPredictor``:

.. code-block:: python

    from deepsort import DeepSortPredictor
    model = DeepSortPredictor(species='human', tissue='Spleen')

Currently we support cell type annotation on human and mouse datasets, with available tissues listed in GitHub Wiki Page (`human <https://github.com/ZJUFanLab/scDeepSort/wiki#human-tissues>`_ and `mouse <https://github.com/ZJUFanLab/scDeepSort/wiki/Mouse_tissues>`_). Note that the first letter of tissues should be in upper case.

Prepare Data
------------

Please refer to `Input Requirement <./input_requirement.html>`_

Evaluate
--------

Once the datasets prepared, users can predict the corresponding cell type (and subtype if exists) for cells. Our predict function supports processing single file in one pass as following:

.. code-block:: python

    test_file = 'test/human/human_Spleen11081_data.csv'
    predictor.predict(test_file, save_path='results')

This method saves results to specific path if provided as keyword argument.

The default setting on hyper-parameters enables scDeepSort to perform reasonably well across all datasets. Please refer to `API Reference <./api_reference.html>`_ for the meaning of different input arguments.

2. Train Your Own Model
***********************
Below is the full script on using scDeepSort for classification on a demo dataset.

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

Our ``DeepSortClassifier`` model takes a list of tuples of file paths as inputs to fit on multiple datasets.

Users are required to prepare the data file and the corresponding cell type file for training and testing as expected in `Input Requirement <./input_requirement.html>`_.


