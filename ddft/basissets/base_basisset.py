import torch
import xitorch as xt
from abc import abstractmethod, abstractproperty

class BaseAtomicBasis(object):
    @abstractproperty
    def dtype(self):
        pass

    @abstractproperty
    def device(self):
        pass

    @staticmethod
    @abstractmethod
    def construct_hamiltonian(grid, bases_list, atomposs):
        pass
