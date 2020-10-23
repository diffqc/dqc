import torch
import numpy as np
from ddft.eks.base_eks import BaseEKS

__all__ = ["Hartree"]

class Hartree(BaseEKS):
    def forward(self, density, gradn=None):
        # density: (nbatch, nr)
        vks = self.potential(density, gradn)
        # dirichlet boundary
        # vks = vks - vks[:,-1]
        eks = 0.5 * vks * density
        return eks

    def potential(self, density, gradn=None):
        vks = self.grid.solve_poisson(-4.0*np.pi*density)
        return vks

    def getfwdparamnames(self, prefix=""):
        return self.grid.getparamnames("solve_poisson", prefix=prefix+"grid.")

    def getparamnames(self, methodname, prefix=""):
        return self.getfwdparamnames(prefix=prefix)
