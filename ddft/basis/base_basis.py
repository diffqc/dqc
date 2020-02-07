from abc import abstractmethod
import torch
import lintorch as lt

class BaseBasis(object):
    @abstractmethod
    def nbasis(self):
        """
        Returns the number of basis.
        """
        pass

    @abstractmethod
    def kinetics(self):
        """
        Returns a lintorch.Module for the Kinetics operator.
        """
        pass

    @abstractmethod
    def vcoulomb(self):
        """
        Returns a lintorch.Module for the Coulomb potential operator with Z=1.
        """
        pass

    @abstractmethod
    def vpot(self, vext):
        """
        Returns a lintorch.Module for the given external potential operator.

        Arguments
        ---------
        * vext: torch.tensor (nbatch, nr)
        """
        pass

    @abstractmethod
    def overlap(self):
        """
        Returns the basis overlap operator in lintorch.Module.
        If the basis are orthogonal, then it should return an identity operator.
        """
        pass

    @abstractmethod
    def tocoeff(self, wfr):
        """
        Obtain the coefficients of the basis from the given input function in
        the spatial grid domain.

        Arguments
        ---------
        * wfr: torch.tensor (nbatch, nr)
            The input argument in spatial grid to be converted to the basis.

        Returns
        -------
        * coeffs: torch.tensor (nbatch, ns)
            The coefficients in the basis.
        """
        pass

    @abstractmethod
    def frocoeff(self, coeffs):
        """
        Transform back from coefficients to the function in spatial domain.

        Arguments
        ---------
        * coeffs: torch.tensor (nbatch, ns)
            The coefficients in the basis.

        Returns
        -------
        * wfr: torch.tensor (nbatch, nr)
            The function in spatial grid domain.
        """
        pass
