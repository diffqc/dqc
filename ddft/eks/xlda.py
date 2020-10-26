import torch
from ddft.utils.safeops import safepow
from ddft.eks.base_eks import BaseEKS

__all__ = ["xLDA"]

class xLDA(BaseEKS):
    # TODO: implement the proper spin polarized case

    def __init__(self, a=-0.7385587663820223, p=4./3):
        self.a = a
        self.p = p

    def forward(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        density = density_up + density_dn
        return self.a * safepow(density.abs(), self.p)

    def potential(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        density = density_up + density_dn
        pot = self.p * self.a * safepow(density.abs(), self.p - 1)
        return pot, pot

    def getfwdparamnames(self, prefix=""):
        return []
