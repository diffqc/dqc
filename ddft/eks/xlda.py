import torch
from ddft.utils.safeops import safepow
from ddft.eks.x import Exchange

__all__ = ["xLDA"]

class xLDA(Exchange):
    def __init__(self, a=-0.7385587663820223, p=4./3):
        self.a = torch.as_tensor(a)
        self.p = torch.as_tensor(p)

    def _forward(self, density, gradn=None):
        return self.a * safepow(density.abs(), self.p)

    def _potential(self, density, gradn=None):
        return self.p * self.a * safepow(density.abs(), self.p - 1)

    def getfwdparamnames(self, prefix=""):
        return [prefix + "a", prefix + "p"]
