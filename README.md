# DQC: Differentiable Quantum Chemistry

![Build](https://img.shields.io/github/workflow/status/diffqc/dqc/ci?style=flat-square)
[![Code coverage](https://img.shields.io/codecov/c/github/diffqc/dqc?style=flat-square)](https://app.codecov.io/gh/diffqc/dqc)
[![Docs](https://img.shields.io/readthedocs/dqc?style=flat-square)](https://dqc.readthedocs.io/)

Differentiable quantum chemistry package.
Currently only support differentiable density functional theory (DFT)
and Hartree-Fock (HF) calculation.

The documentation can be found at: https://dqc.readthedocs.io/

## Requirements

* [python](https://www.python.org) 3.7 or newer with [pip](https://pip.pypa.io/en/stable/installing/)
* [cmake](https://cmake.org/) 3.18 or newer

## Installation

First, you need to install the requirements above.
After you got the requirements, then you can install dqc from terminal by:

    git clone https://github.com/diffqc/dqc
    cd dqc
    python -m pip install -e .
