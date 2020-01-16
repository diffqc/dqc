from abc import abstractmethod, abstractproperty
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
    def kinetics_diag(self, *params):
        """
        Returns the diagonal of the kinetics part of the Hamiltonian.

        Arguments
        ---------
        * *params: list of torch.tensor (nbatch, ...)
            List of parameters that specifies the kinetics of the Hamiltonian.
        """
        pass

    @abstractmethod
    def rgrid(self):
        """
        Returns a tensor which specifies the spatial grid of vext and wf.
        The shape is (nr, ndim) or (nr,) for ndim == 1.
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

    def diag(self, vext, *params):
        return vext + self.kinetics_diag(*params)

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
