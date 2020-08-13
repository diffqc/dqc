from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import lintorch as lt

class BaseHamiltonGenerator(object):
    def __init__(self, shape, dtype=None, device=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def get_hamiltonian(self, vext, *hparams):
        pass

    def get_overlap(self, *sparams):
        pass
