from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import lintorch as lt

class BaseHamiltonGenerator(lt.EditableModule):
    def __init__(self, shape, dtype=None, device=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def get_hamiltonian(self, vext, *hparams):
        pass

    @abstractmethod
    def get_overlap(self, *sparams):
        pass

    @abstractmethod
    def dm2dens(self, dm):
        pass

    @abstractmethod
    def getparamnames(self, methodname, prefix=""):
        pass
