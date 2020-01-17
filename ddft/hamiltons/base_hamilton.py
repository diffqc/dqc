from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
from ddft.modules.base_linear import BaseLinearModule

class BaseHamilton(BaseLinearModule):
    @abstractmethod
    def kinetics(self, wf, *params):
        """
        Compute the kinetics part of the Hamiltonian given the wavefunction, wf.

        Arguments
        ---------
        * wf: torch.tensor (nbatch, nr, ncols)
            The wavefunction with length `nr` each. The value in the tensor
            indicates the value of wavefunction in the spatial grid given in
            self.rgrid.
        * *params: list of torch.tensor (nbatch, ...)
            List of parameters that specifies the kinetics part.

        Returns
        -------
        * k: torch.tensor (nbatch, nr, ncols)
            The kinetics part of the Hamiltonian
        """
        pass

    @abstractmethod
    def kinetics_diag(self, nbatch, *params):
        """
        Returns the diagonal of the kinetics part of the Hamiltonian.

        Arguments
        ---------
        * nbatch: int
            The number of batch of the kinetics matrix to be returned
        * *params: list of torch.tensor (nbatch, ...)
            List of parameters that specifies the kinetics of the Hamiltonian.
        """
        pass

    @abstractproperty
    def rgrid(self):
        """
        Returns a tensor which specifies the spatial grid of vext and wf.
        The shape is (nr, ndim) or (nr,) for ndim == 1.
        """
        pass

    @abstractproperty
    def boxshape(self):
        """
        Returns the box shape.
        """
        pass

    @abstractmethod
    def getdens(self, eigvec2):
        """
        Calculate the density given the square of eigenvectors.
        The density returned should fulfill integral{density * dr} = 1.

        Arguments
        ---------
        * eigvec2: torch.tensor (nbatch, nr, neig)
            The eigenvectors arranged in dimension 1 (i.e. nr). It is assumed
            that eigvec2.sum(dim=1) == 1

        Returns
        -------
        * density: torch.tensor (nbatch, nr, neig)
            The density where the integral over the space should be equal to 1.
        """
        pass

    @abstractmethod
    def integralbox(self, p):
        """
        Perform integral p(r) dr where p is tensor with shape (nbatch,nr)
        describing the value in the box.
        """
        pass

    def diag(self, vext, *params):
        nbatch = vext.shape[0]
        return vext + self.kinetics_diag(nbatch, *params)

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

    @property
    def shape(self):
        """
        Returns the matrix shape of the Hamiltonian.
        """
        if not hasattr(self, "_shape"):
            nr = len(self.rgrid)
            self._shape = (nr, nr)
        return self._shape

    def forward(self, wf, vext, *params):
        """
        Compute the Hamiltonian of a wavefunction and external potential.
        The wf and vext should be located at the spatial grid specified in
        self.rgrid.

        Arguments
        ---------
        * wf: torch.tensor (nbatch, nr) or (nbatch, nr, ncols)
            The wavefunction in spatial domain
        * vext: torch.tensor (nbatch, nr)
            The external potential in spatial domain
        * *params: list of torch.tensor (nbatch, ...)
            List of parameters that specifies the kinetics part.

        Returns
        -------
        * h: torch.tensor (nbatch, nr) or (nbatch, nr, ncols)
            The calculated Hamiltonian
        """
        # wf: (nbatch, nr) or (nbatch, nr, ncols)
        # vext: (nbatch, nr)

        # normalize the shape of wf
        wfndim = wf.ndim
        if wfndim == 2:
            wf = wf.unsqueeze(-1)

        nbatch = wf.shape[0]
        kinetics = self.kinetics(wf, *params)
        extpot = vext.unsqueeze(-1) * wf
        h = kinetics + extpot

        if wfndim == 2:
            h = h.squeeze(-1)
        return h
