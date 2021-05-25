.. dqc documentation master file, created by
   sphinx-quickstart on Mon May 03 15:44:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DQC: differentiable quantum chemistry
=====================================

DQC is a PyTorch-based quantum chemistry simulation software that can
automatically provide gradients of (almost) any variables with respect to
(almost) any variables.
DQC provides analytic first and higher order derivatives automatically
using PyTorch's autograd engine.

DQC's example API:

.. doctest:: python

    >>> import torch
    >>> import dqc
    >>> atomzs, atomposs = dqc.parse_moldesc("H -1 0 0; H 1 0 0")
    >>> atomposs = atomposs.requires_grad_()  # mark atomposs as differentiable
    >>> mol = dqc.Mol((atomzs, atomposs), basis="3-21G")
    >>> qc = dqc.HF(mol).run()
    >>> ene = qc.energy()  # calculate the energy
    >>> force = -torch.autograd.grad(ene, atomposs)[0]  # calculate the force

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getstart/installation
   getstart/contribute
   getstart/tutorials

.. toctree::
   :maxdepth: 1
   :caption: Modules

   api/dqc_system/index
   api/dqc_qccalc/index
   api/dqc_api/index

.. toctree::
   :maxdepth: 1
   :caption: Submodules

   api/dqc_hamilton/index
   api/dqc_xc/index
   api/dqc_utils/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
