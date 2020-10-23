import torch
from ddft.utils.safeops import safepow
from ddft.eks.base_eks import BaseEKS

__all__ = ["xLDA"]

class xLDA(BaseEKS):
    def forward(self, density, gradn=None):
        return -0.7385587663820223 * safepow(density.abs(), 4./3)

    def getfwdparamnames(self, prefix=""):
        return []
