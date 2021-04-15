Get Started
==================

This is a quick start guide for you to try out deep forest. The full script is available at :ref:`script`.

Installation
------------

The package is available on GitHub.

.. code-block:: bash

    $ git clone https://github.com/ZJUFanLab/scDeepSort
    $ python setup.py

Load Data
---------

scDeepSort assumes data to be in the form of 2D Numpy array of shape (``n_samples``, ``n_features``). It will conduct internal check and transformation on the input data. For example, the code snippet below loads a toy dataset on cell type annotation:

.. code-block:: python

    x, y = load_data()

Define the Model
----------------

scDeepSort provides unified APIs on binary classification and multi-class classification. For the demo dataset on classification, the corresponding model is ``CascadeForestClassifier``:

.. code-block:: python

    from scdeepsort import scDeepSort
    model = scDeepSort()

A key advantage of deep forest is its **adaptive model complexity depending on the dataset**. The default setting on hyper-parameters enables it to perform reasonably well across all datasets. Please refer to `API Reference <./api_reference.html>`__ for the meaning of different input arguments.

Train and Evaluate
------------------

scDeepSort provides Scikit-Learn like APIs on training and evaluating. Given the training data ``X_train`` and labels ``y_train``, the training stage is triggered with the following code snippet:

.. code-block:: python

    model.fit(X_train, y_train)

Once the model was trained, you can call :meth:`predict` to produce prediction results on the testing data ``X_test``.

.. code-block:: python

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100  # classification accuracy

Save and Load
-------------

scDeepSort also provides easy-to-use APIs on model serialization. Here, ``MODE_DIR`` is the directory to save the model.

.. code-block:: python

    model.save(MODEL_DIR)

Given the saving results, you can call :meth:`load` to use deep forest for prediction:

.. code-block:: python

    new_model = CascadeForestClassifier()
    new_model.load(MODEL_DIR)

Notice that :obj:`new_model` is not the same as :obj:`model`, because only key information used for model inference was saved.

.. _script:

Example
-------

Below is the full script on using deep forest for classification on a demo dataset.

.. code-block:: python

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

