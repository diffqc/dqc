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

.. code-block:: python

    import torch
    from dqc import Mol, HF
    # set up the H2 molecule, forcing the atom positions to be differentiable
    mol = Mol("H -1 0 0; H 1 0 0", basis="3-21G", diffparams=["atompos"])
    qc = HF(mol).run()
    ene = qc.energy()  # calculate the energy
    force = -torch.autograd.grad(ene, mol.atompos)[0]  # calculate the force


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getstart/installation
   getstart/contribute


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
