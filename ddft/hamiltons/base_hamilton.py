from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import lintorch as lt

class BaseHamilton(lt.Module):
    ################################ Basis part ################################
    @abstractmethod
    def forward(self, wf, vext, *params):
        """
        Compute the Hamiltonian of a wavefunction in the basis domain
        and external potential in the spatial domain.
        The wf is located in the basis domain of the Hamiltonian and vext in
        the spatial domain (rgrid).
        The interpretation of boundary condition and interpolation between
        points in the spatial domain depends on the chosen space.

        Arguments
        ---------
        * wf: torch.tensor (nbatch, ns, ncols)
            The wavefunction in the basis domain
        * vext: torch.tensor (nbatch, nr)
            The external potential in spatial domain. This should be the total
            potential the non-interacting particles feel
            (i.e. xc, Hartree, and external).
        * *params: list of torch.tensor (nbatch, ...)
            List of parameters that specifies the kinetics part.

        Returns
        -------
        * h: torch.tensor (nbatch, ns, ncols)
            The calculated Hamiltonian
        """
        pass

    @abstractmethod
    def precond(self, y, vext, *params, biases=None):
        """
        Apply the preconditioning of the Hamiltonian to the tensor `y`.
        The return shape: (nbatch, ns, ncols)

        Arguments
        ---------
        * y: torch.tensor (nbatch, ns, ncols)
            The tensor where the preconditioning is applied
        * vext: torch.tensor (nbatch, nr)
            The external potential in spatial domain. This should be the total
            potential the non-interacting particles feel
            (i.e. xc, Hartree, and external).
        * *params: list of torch.tensor (nbatch, ...)
            The list of parameters that define the Hamiltonian
        * biases: torch.tensor (nbatch, ncols) or None
            If not None, it will compute the preconditioning for (H-b*I) for
            each column of y. If None, then it is zero.

        Returns
        -------
        * x: torch.tensor (nbatch, ns, ncols)
            The output of the preconditioning.
        """
        pass

    ################################ Grid part ################################
    @abstractproperty
    def rgrid(self):
        """
        The spatial grid with shape (nr, ndim).
        """
        pass

    @abstractmethod
    def getvhartree(self, dens):
        """
        Return the Hartree potential in spatial grid given the density in
        spatial grid.

        Arguments
        ---------
        * dens: torch.tensor (nbatch, nr)
            The density profile in spatial grid.

        Returns
        -------
        * vhartree: torch.tensor (nbatch, nr)
            The Hartree potential.
        """
        pass

    @abstractmethod
    def getdens(self, eigvec):
        """
        Calculate the density given the eigenvectors.
        The density returned should fulfill integral{density * dr} = 1.

        Arguments
        ---------
        * eigvec: torch.tensor (nbatch, ns, neig)
            The eigenvectors arranged in dimension 1 (i.e. ns). It is assumed
            that eigvec.sum(dim=1) == 1.

        Returns
        -------
        * density: torch.tensor (nbatch, nr, neig)
            The density where the integral over the space should be equal to 1.
            The density is in the spatial domain.
        """
        pass

    @abstractmethod
    def integralbox(self, p, dim=-1):
        """
        Perform integral p(r) dr where p is tensor with shape (nbatch,nr)
        describing the value in the box.
        """
        pass
