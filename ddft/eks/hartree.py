import torch
import numpy as np
from ddft.eks.base_eks import BaseEKS

__all__ = ["Hartree"]

class Hartree(BaseEKS):
    def __init__(self, grid):
        super(Hartree, self).__init__()
        self.grid = grid

    def forward(self, density):
        # density: (nbatch, nr)
        vks = self.grid.solve_poisson(-4.0*np.pi*density)
        # dirichlet boundary
        vks = vks - vks[:,-1]
        eks = 0.5 * vks * density
        return eks
