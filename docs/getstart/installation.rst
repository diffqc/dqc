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

    git clone --recursive https://github.com/mfkasim1/dqc
    cd dqc
    git submodule sync
    git submodule update --init --recursive
    python -m pip install -e .

Then, to install the C-libraries, type:

.. code-block::

    # installing libraries from PySCF
    cd lib
    mkdir build; cd build
    cmake ..
    make

    # installing libcint
    cd ../libcint
    mkdir build; cd build
    cmake ..
    make

    # installing libxc
    cd ../../../submodules/libxc/
    python setup.py install
