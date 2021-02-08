# DQC: Differentiable Quantum Chemistry

Differentiable quantum chemistry package.
Currently only support differentiable density functional theory (DFT) calculation.

## Requirements

* [python](https://www.python.org) 3.6 or newer
* [pip](https://pip.pypa.io/en/stable/installing/)
* [pytorch](https://pytorch.org) 1.7.1 or newer (I used the nightly build which is for
  version 1.8.0, but I think 1.7.1 should work as well)
* [cmake](https://cmake.org/) 2.8 or newer

## Installation

First, you need to install the requirements above.
After you got the requirements, then you can install dqc from terminal by:

    git clone --recursive https://github.com/mfkasim1/dqc
    cd dqc
    git submodule sync
    git submodule update --init --recursive

Then, install the libraries by:

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
