Alchemical perturbation
=======================

In this tutorial, we will show how to estimate properties of molecules using
alchemical perturbation.
Specifically, we will estimate the distance between two atoms in
:math:`\mathrm{CO}` and :math:`\mathrm{BF}` molecules by only calculating the
properties of :math:`\mathrm{N_2}` and its alchemical perturbation.

Let's denote the atomic number of the atoms in the diatomic molecule as

.. math::
    Z_{\pm}(\lambda) = 7 \pm \lambda

parameterized by a variable :math:`\lambda`.
One of the atom takes the plus sign while another one takes the minus sign to
keep the number of electrons constant.

The equilibrium distance between the atoms is defined as

.. math::

    s^*(\lambda) = \arg\min_s \mathcal{E}(s, \lambda)

where :math:`\mathcal{E}` is the total energy as a function of the atomic
distance :math:`s` and :math:`\lambda`.
What we will do is to estimate the equilibrium distance for :math:`\lambda = 1`
(for :math:`\mathrm{CO}`) and :math:`\lambda = 2` (for :math:`\mathrm{BF}`)
using Taylor expansion,

.. math::

    s^*(\lambda) \approx s^*(0) + \lambda \frac{\partial s^*}{\partial \lambda} +
      \frac{1}{2} \lambda^2 \frac{\partial^2 s^*}{\partial \lambda^2}

As a demonstration, we will use Hartree-Fock calculation with 3-21G basis set.
First, we need to import modules and set up variables that we will need for the
calculations.

.. doctest::

    >>> import torch
    >>> import dqc
    >>> import xitorch.optimize  # for differentiable optimization
    >>> dtype = torch.double
    >>> basis = dqc.loadbasis("7:3-21G")

``xitorch`` is a great library that provides differentiable functionals that
we will use in this tutorial.
The last line with :meth:`dqc.loadbasis` loads the basis 3-21G for atomic
number 7. We will use the same basis for all values of :math:`\lambda` to make
sure there is no discontinuity in the properties.

Next, we need to define a function that calculates the energy given the distance
:math:`s` and :math:`\lambda`.

.. doctest::

    >>> def get_energy(s, lmbda):
    ...     atomzs = 7.0 + torch.tensor([1.0, -1.0], dtype=dtype) * lmbda
    ...     atomposs = torch.tensor([[-0.5, 0, 0], [0.5, 0, 0]], dtype=dtype) * s
    ...     mol = dqc.Mol((atomzs, atomposs), spin=0, basis=[basis, basis])
    ...     qc = dqc.HF(mol).run()
    ...     return qc.energy()

Once the function is defined, then we can calculate the equilibrium distance
for :math:`\mathrm{N_2}` molecule.

.. doctest::

    >>> lmbda = torch.tensor(0.0, dtype=dtype).requires_grad_()
    >>> s0_n2 = torch.tensor(2.04, dtype=dtype)  # initial guess of the distance
    >>> smin_n2 = xitorch.optimize.minimize(
    ...     get_energy, s0_n2, params=(lmbda,), method="gd", step=1e-2)
    >>> print(smin_n2)
    tensor(2.0460, dtype=torch.float64, grad_fn=<_RootFinderBackward>)

``xitorch.optimize.minimize`` finds the parameters ``s`` that minimizes the
energy given the parameters ``lmbda``.
The output of ``xitorch.optimize.minimize`` is now differentiable with respect
to the parameter ``lmbda``.

.. doctest::

    >>> grad_lmbda = torch.autograd.grad(smin_n2, lmbda, create_graph=True)[0]
    >>> grad2_lmbda = torch.autograd.grad(grad_lmbda, lmbda, create_graph=True)[0]
    >>> print(grad_lmbda.detach(), grad2_lmbda.detach())
    tensor(-2.0242e-10, dtype=torch.float64) tensor(0.1323, dtype=torch.float64)

Now, we can estimate the equilibrium distance of :math:`\mathrm{CO}` and
:math:`\mathrm{BF}`,

.. doctest::

    >>> smin_co = smin_n2 + grad_lmbda + 0.5 * grad2_lmbda
    >>> smin_bf = smin_n2 + grad_lmbda * 2 + 0.5 * grad2_lmbda * 2 ** 2
    >>> print(smin_co.detach(), smin_bf.detach())
    tensor(2.1121, dtype=torch.float64) tensor(2.3106, dtype=torch.float64)

For reference, the equilibrium distances for :math:`\mathrm{CO}` and
:math:`\mathrm{BF}` by minimizing the energy are 2.1119 and 2.3103 Bohr,
respectively, which are quite close to the values above.
