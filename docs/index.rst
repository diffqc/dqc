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

.. code-block:: python

    from dqc import Mol, HF
    mol = Mol("H -1 0 0; H 1 0 0", basis="3-21G")
    qc = HF(mol).run()
    ene = qc.energy()  # calculate the energy


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
