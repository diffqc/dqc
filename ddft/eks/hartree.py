import torch
import numpy as np
from ddft.eks.base_eks import BaseEKS

__all__ = ["Hartree"]

class Hartree(BaseEKS):
    def forward(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        # density: (nbatch, nr)
        vks, _ = self.potential(density_up, density_dn, gradn_up, gradn_dn)
        # dirichlet boundary
        # vks = vks - vks[:,-1]
        eks = 0.5 * vks * (density_up + density_dn)
        return eks

    def potential(self, density_up, density_dn,
                  gradn_up=None, gradn_dn=None):
        vks = self.grid.solve_poisson(-4.0 * np.pi * (density_up + density_dn))
        return vks, vks

    def getfwdparamnames(self, prefix=""):
        return self.grid.getparamnames("solve_poisson", prefix=prefix+"grid.")

    def getparamnames(self, methodname, prefix=""):
        return self.getfwdparamnames(prefix=prefix)
