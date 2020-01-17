from abc import abstractmethod, abstractproperty
from functools import reduce
import torch

"""
Space is class that regulates the transformation from spatial domain to another
domain that is invertable to the spatial domain.
The purpose of transforming signal to other domain is to ease some calculations,
such as the use of frequency space.
"""

class BaseSpace(object):
    @abstractproperty
    def rgrid(self):
        """
        Returns a tensor which specifies the spatial grid of the signal.
        The shape is (nr,ndim) or (nr,ndim,2) for complex grid.
        """
        pass

    @abstractproperty
    def qgrid(self):
        """
        Returns a tensor specifying the grid on transformed domain.
        The shape is (ns,ndim) or (ns,ndim,2) for complex grid.
        """
        pass

    @abstractproperty
    def boxshape(self):
        """
        Returns the box shape in spatial domain.
        """
        pass

    @abstractproperty
    def qboxshape(self):
        """
        Returns the box shape in the transformed domain.
        """
        pass

    @abstractmethod
    def transformsig(self, sig, dim=-1):
        """
        If the Hamiltonian works in non-spatial domain, then this method
        should transform signal from spatial to the intended domain.
        The signal should be flatten in the given dimension.

        Arguments
        ---------
        * sig: torch.tensor (...,nr,...) or (...,nr,...,2) for complex
            The signal to be transformed from spatial domain to the other
            domain.
        * dim: int
            The dimension where the signal nr is located.

        Returns
        -------
        * tsig: torch.tensor (...,ns,...) or (...,ns,...,2) for complex
            The transformed signal.
        """
        pass

    @abstractmethod
    def invtransformsig(self, tsig, dim=-1):
        """
        If the Hamiltonian works in non-spatial domain, then this method
        should transform signal from the Hamiltonian domain to the spatial
        domain.
        The signal should be flatten in the given dimension.

        Arguments
        ---------
        * tsig: torch.tensor (...,ns,...) or (...,ns,...,2) for complex
            The signal in the transformed domain.
        * dim: int
            The dimension where the signal ns is located.

        Returns
        -------
        * sig: torch.tensor (...,nr,...) or (...,nr,...,2) for complex
            The signal in the spatial domain.
        """
        pass

    def flattensig(self, sig, dim=-1, qdom=False):
        """
        Flatten the signal whose shape is (...,*boxshape,...) into
        (...,nr,...).

        Arguments
        ---------
        * sig: torch.tensor
            The signal to be flatten
        * dim: int
            The dimension position where the `nr` will be located at the output.
        * qdom: bool
            Flag to indicate whether the signal is in qdomain.
        """
        ndim = sig.ndim
        boxshape = self.qboxshape if qdom else self.boxshape
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

    def boxifysig(self, sig, dim=-1, qdom=False):
        """
        Shape the signal into box shape, i.e. transform from (...,nr,...) into
        (...,*boxshape,...)

        Arguments
        ---------
        * sig: torch.tensor
            The signal to be boxify
        * dim: int
            The dimension where the `nr` is located.
        * qdom: bool
            Flag to indicate whether the signal is in qdomain.
        """
        # make dim positive
        ndim = sig.ndim
        boxshape = self.qboxshape if qdom else self.boxshape
        if dim < 0:
            dim = ndim + dim

        sigshape = sig.shape
        shapel = [sigshape[i] for i in range(dim)]
        shaper = [sigshape[i] for i in range(dim+1, ndim)]
        newshape = shapel + [*boxshape] + shaper
        return sig.view(*newshape)
