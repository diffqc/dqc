import torch
from ddft.utils.safeops import safepow
from ddft.eks.base_eks import BaseEKS

__all__ = ["xLDA"]

class xLDA(BaseEKS):
    def forward(self, density):
        return -0.7385587663820223 * safepow(density.abs(), 4./3)
