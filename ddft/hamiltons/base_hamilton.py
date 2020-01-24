from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
from ddft.modules.base_linear import BaseLinearModule

class BaseHamilton(BaseLinearModule):
    @abstractmethod
    def apply(self, wf, vext, *params):
        """
        Compute the Hamiltonian of a wavefunction in the chosen domain
        and external potential in the spatial domain.
        The wf is located in the chosen domain of the Hamiltonian (having the
        same shape as qgrid) and vext in the spatial domain (rgrid).

        Arguments
        ---------
        * wf: torch.tensor (nbatch, ns, ncols)
            The wavefunction in the chosen domain
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

    def applyT(self, wfr, vext, *params):
        raise RuntimeError("The tranposed function for class %s is not implemented." %\
            (self.__class__.__name__))

    @abstractmethod
    def getdens(self, eigvec):
        """
        Calculate the density given the of eigenvectors in the chosen domain.
        The density returned should fulfill integral{density * dr} = 1.

        Arguments
        ---------
        * eigvec: torch.tensor (nbatch, ns, neig)
            The eigenvectors arranged in dimension 1 (i.e. ns). It is assumed
            that eigvec.sum(dim=1) == 1.
            The eigenvectors are in the chosen domain.

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

    @abstractproperty
    def shape(self):
        """
        Returns the matrix shape of the Hamiltonian.
        """
        return

    @abstractmethod
    def diag(self, vext, *params):
        """
        Returns the diagonal of the matrix for each batch.
        The return shape: (nbatch, ns)
        """
        pass

    ########################### implemented methods ###########################
    def flattensig(self, sig, dim=-1):
        """
        Flatten the signal whose shape is (...,*boxshape,...) into
        (...,nr,...).

        Arguments
        ---------
        * sig: torch.tensor
            The signal to be flatten
        * dim: int
            The dimension position where the `nr` will be located at the output.
        """
        ndim = sig.ndim
        boxshape = self.boxshape
        nboxshape = len(boxshape)
        nr = reduce(lambda x,y: x*y, boxshape)

        # get the dim into starting dim
        if dim < 0:
            dim = ndim + dim - (nboxshape - 1)

        sigshape = sig.shape
        shapel = [sigshape[i] for i in range(dim)]
        shaper = [sigshape[i] for i in range(dim+nboxshape, ndim)]
        newshape = shapel + [nr] + shaper
        return sig.view(*newshape)

    def boxifysig(self, sig, dim=-1):
        """
        Shape the signal into box shape, i.e. transform from (...,nr,...) into
        (...,*boxshape,...)

        Arguments
        ---------
        * sig: torch.tensor
            The signal to be boxify
        * dim: int
            The dimension where the `nr` is located.
        """
        # make dim positive
        ndim = sig.ndim
        if dim < 0:
            dim = ndim + dim

        sigshape = sig.shape
        shapel = [sigshape[i] for i in range(dim)]
        shaper = [sigshape[i] for i in range(dim+1, ndim)]
        newshape = shapel + [*self.boxshape] + shaper
        return sig.view(*newshape)

    def forward(self, wf, vext, *params):
        # wf: (nbatch, nr) or (nbatch, nr, ncols)
        # vext: (nbatch, nr)

        # normalize the shape of wf
        wfndim = wf.ndim
        if wfndim == 2:
            wf = wf.unsqueeze(-1)

        h = self.apply(wf, vext, *params)

        if wfndim == 2:
            h = h.squeeze(-1)
        return h

    def transpose(self, wf, vext, *params):
        if self.issymmetric:
            return self.forward(wf, vext, *params)
        else:
            wfndim = wf.ndim
            if wfndim == 2:
                wf = wf.unsqueeze(-1)
            hT = self.applyT(wf, vext, *params)
            if wfndim == 2:
                hT = hT.squeeze(-1)
            return hT
