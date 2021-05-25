Calculating force
=================

Calculating force in DQC is straightforward as shown by the code below

.. doctest::

    >>> import torch
    >>> import dqc
    >>> atomzs = torch.tensor([1, 1])
    >>> atomposs = torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=torch.double).requires_grad_()
    >>> mol = dqc.Mol(moldesc=(atomzs, atomposs), basis="3-21G")
    >>> qc = dqc.HF(mol).run()
    >>> ene = qc.energy()  # calculate the energy
    >>> force = -torch.autograd.grad(ene, atomposs)[0]  # calculate the force

Let's take a look at each parts.
First, we set up the atomic numbers and positions with lines:

.. doctest::

    >>> atomzs = torch.tensor([1, 1])
    >>> atomposs = torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=torch.double).requires_grad_()

The system has two atoms with atomic number 1 and 1
(i.e. hydrogen molecule) as set by ``atomzs`` variable.
In the next line, ``atomposs`` describes the position of each atom, i.e. the first
one is at ``(1, 0, 0)`` and the second one is at ``(-1, 0, 0)``.
Please note that ``atomposs`` is marked as differentiable by ``.requires_grad_()``
command.
This is required as we want to differentiate the energy later with respect to the
atomic positions.

Next, we construct the DQC system by:

.. doctest::

    >>> mol = dqc.Mol((atomzs, atomposs), basis="3-21G")

The first argument of :class:`~dqc.Mol` is molecular description which can accept
a tuple of ``(atomzs, atomposs)`` or a string description (explained later).
The ``basis`` keyword choose the basis for each atom.
In this case, it uses ``"3-21G"`` basis set.

Once the DQC system is constructed, then we can run the calculation by

.. doctest::

    >>> qc = dqc.HF(mol).run()
    >>> ene = qc.energy()  # calculate the energy

The first line above runs the simulation until it reaches convergence.
Then, the next line calculates the energy.
The output energy is differentiable with respect to floating point tensors
in the system that are set to be differentiable.
Therefore, the force can be simply calculated by

.. doctest::

    >>> force = -torch.autograd.grad(ene, mol.atompos)[0]  # calculate the force

How if we have molecule description in string, e.g. ``"H -1 0 0; H 1 0 0"``?
In this case, we need a help from :meth:`~dqc.parse_moldesc`,

.. doctest::

    >>> import torch
    >>> import dqc
    >>> atomzs, atomposs = dqc.parse_moldesc("H -1 0 0; H 1 0 0")
    >>> atomposs = atomposs.requires_grad_()  # marking atomposs as differentiable
    >>> mol = dqc.Mol(moldesc=(atomzs, atomposs), basis="3-21G")
    >>> qc = dqc.HF(mol).run()
    >>> ene = qc.energy()  # calculate the energy
    >>> force = -torch.autograd.grad(ene, atomposs)[0]  # calculate the force

The only difference in this case is the lines

.. doctest::

    >>> atomzs, atomposs = dqc.parse_moldesc("H -1 0 0; H 1 0 0")
    >>> atomposs = atomposs.requires_grad_()  # marking atomposs as differentiable

where :meth:`~dqc.parse_moldesc` parses the string and returns two tensors describing
the atomic numbers and atomic positions.
The rest are just the same as the previous case.
