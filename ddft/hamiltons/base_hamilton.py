from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import lintorch as lt

class BaseHamilton(lt.Module):
    # TODO: do initialization to check if the methods are implemented properly

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
    def precond(self, y, vext, *params, biases=None, M=None, mparams=None):
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
        * M: lintorch.Module or None
            The transformation on the biases side. If biases is None,
            then this argument is ignored. If None or ignored, then M=I.
        * mparams: list of differentiable torch.tensor
            List of differentiable torch.tensor to be put to M.

        Returns
        -------
        * x: torch.tensor (nbatch, ns, ncols)
            The output of the preconditioning.
        """
        pass

    @property
    def overlap(self):
        """
        overlap should act as a function that represents the overlap
        matrix of the basis (i.e. F^T*F where F is the basis).

        If the basis is orthonormal, then it should remain as None property.
        Otherwise, it should be wrapped with @lintorch.module decorator
        and takes an input of (wf, *rparams).
        """
        return None

    @abstractmethod
    def tocoeff(self, wfr, dim=-2):
        """
        Convert the signal in spatial domain to the coefficients of the basis
        in the given dimension.

        Arguments
        ---------
        * wfr: torch.tensor (..., nr, ...)
            The signal in spatial domain.

        Returns
        -------
        * wfs: torch.tensor (..., ns, ...)
            The coefficients of the basis of the signal.
        """
        pass

    @abstractmethod
    def torgrid(self, wfs, dim=-2):
        """
        Obtain the signal in spatial domain from the coefficients of the basis
        in the given dimension.

        Arguments
        ---------
        * wfs: torch.tensor (..., ns, ...)
            The coefficients of the basis of the signal.

        Returns
        -------
        * wfr: torch.tensor (..., nr, ...)
            The signal in spatial domain.
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
        eigvec_r = self.torgrid(eigvec, dim=-2)
        dens = (eigvec_r * eigvec_r)
        sumdens = self.integralbox(dens, dim=1).unsqueeze(1)
        return dens / sumdens

    @abstractmethod
    def integralbox(self, p, dim=-1):
        """
        Perform integral p(r) dr where p is tensor with shape (nbatch,nr)
        describing the value in the box.
        """
        pass
