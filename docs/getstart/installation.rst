Installation
============

Requirements
------------

* python >= 3.7
* pytorch >= 1.8 (install `here <https://pytorch.org/>`_)
* cmake >= 3.18

Installation
------------

As we have no released version, only source installation is possible.
To do this, type in your terminal:

.. code-block::

    git clone --recursive https://github.com/diffqc/dqc
    cd dqc
    python -m pip install -e .
    python setup.py build_ext
