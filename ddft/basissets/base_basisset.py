import torch
from abc import abstractmethod, abstractproperty

class BaseBasisModule(torch.nn.Module):
    @abstractproperty
    def dtype(self):
        pass

    @abstractproperty
    def device(self):
        pass

    @abstractmethod
    def construct_basis(self, atomzs, atomposs):
        """
        Construct the basis and store them as parameters of the module.
        Calling this function will reset the basis parameters.
        """
        pass

    @abstractmethod
    def get_hamiltonian(self, grid):
        """
        Returns the hamiltonian using the basis parameters set up earlier with
        construct_basis.
        """
        pass
