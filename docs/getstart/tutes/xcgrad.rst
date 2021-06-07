Calculating gradients of xc functional
======================================

One of DQC applications is to optimize xc functional to fit some properties.
In this tutorial, we will see how to get the gradient of custom xc functionals
for the optimization of custom xc functionals.

First, we need to define our custom xc functional.

.. jupyter-execute::

    import torch
    import dqc
    import dqc.xc
    import dqc.utils
    class MyLDAX(dqc.xc.CustomXC):
        def __init__(self, a, p):
            super().__init__()
            self.a = a
            self.p = p

        @property
        def family(self):
            # 1 for LDA, 2 for GGA, 4 for MGGA
            return 1

        def get_edensityxc(self, densinfo):
            # densinfo has up and down components
            if isinstance(densinfo, dqc.utils.SpinParam):
                # spin-scaling of the exchange energy
                return 0.5 * (self.get_edensityxc(densinfo.u * 2) + self.get_edensityxc(densinfo.d * 2))
            else:
                rho = densinfo.value.abs() + 1e-15  # safeguarding from nan
                return self.a * rho ** self.p

    a = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.double))
    p = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.double))
    myxc = MyLDAX(a, p)

The base class :class:`~CustomXC` is required to define a new xc functional.
:class:`~CustomXC` is a child class of ``torch.nn.Module``, so the initial
``super().__init__()`` is required.
In our custom xc functional, only ``get_edensityxc`` that needs to be written,
which calculates the xc energy density per volume, as well as specifying the
family of the functional.

The ``densinfo`` input of ``get_edensityxc`` can be either: :class:`~dqc.utils.SpinParam`
or :class:`~dqc.utils.ValGrad`.
:class:`~dqc.utils.SpinParam` is DQC data structure to store variables for spin
up and spin down.
:class:`~dqc.utils.ValGrad` is another DQC data structure to save the density
information by having attributes: ``value`` for local value, ``grad`` for local
gradients, ``lapl`` for the local laplacian, and ``kin`` for the local kinetic
energy.

Once the custom xc functional is defined, we can use it for DFT calculation.

.. jupyter-execute::

    mol = dqc.Mol(moldesc="H -1 0 0; H 1 0 0", basis="3-21G")
    qc = dqc.KS(mol, xc=myxc).run()
    ene = qc.energy()
    print(ene)

And to get the gradient with respect to the xc parameters, it is straightforward.

.. jupyter-execute::

    grad_a, grad_p = torch.autograd.grad(ene, (a, p))
    print(grad_a, grad_p)
