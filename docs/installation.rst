Installation Guide
==================

Stable Version
--------------

scDeepSort requires Python version 3.7, 3.8 or 3.9. 

For the convenience of users, we provide CPU and CUDA builds in different compressed files. These builds share the same package name.

For users with GPU support, please first check out your CUDA version by running the following command:

.. code-block:: bash

    $ nvcc --version | grep release

For users with a CUDA 10.2 build, please download `scDeepSort-v1.0-cu102.tar.gz <https://github.com/ZJUFanLab/scDeepSort/releases>`_ from our GitHub `release page <https://github.com/ZJUFanLab/scDeepSort/releases>`_.

The package is then available using:

.. code-block:: bash

    $ pip install scDeepSort-v1.0-cu102.tar.gz

The package is used a few package dependencies which will be handled automatically. It is recommended to use the package environment from `Anaconda <https://www.anaconda.com/>`__ since it already installs all required packages.

Notice that only the 64-bit Linux are officially supported.

Quick Start
------------

To verify your installation, you can run the following test code.

.. code-block:: python

    import deepsort
    deepsort.demo()

The program will end with "Test successfully!" if everything is good.

