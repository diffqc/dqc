import torch
import numpy as np
from ddft.eks.base_eks import BaseEKS

__all__ = ["Hartree"]

class Hartree(BaseEKS):
    def forward(self, densinfo_u, densinfo_d):
        # density: (nbatch, nr)
        density = densinfo_u.density + densinfo_d.density
        vks, _ = self.potential(densinfo_u, densinfo_d)
        # dirichlet boundary
        # vks = vks - vks[:,-1]
        eks = 0.5 * vks * density
        return eks

    def potential(self, densinfo_u, densinfo_d):
        density = densinfo_u.density + densinfo_d.density
        vks = self.grid.solve_poisson(-4.0 * np.pi * density)
        return vks, vks

    def getfwdparamnames(self, prefix=""):
        return self.grid.getparamnames("solve_poisson", prefix=prefix+"grid.")

    def getparamnames(self, methodname, prefix=""):
        return self.getfwdparamnames(prefix=prefix)
