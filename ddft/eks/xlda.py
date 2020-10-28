import torch
from ddft.utils.safeops import safepow
from ddft.eks.x import Exchange

__all__ = ["xLDA"]

class xLDA(Exchange):
    def __init__(self):
        self.a = torch.tensor(-0.7385587663820223)
        self.p = torch.tensor(4./3)

    def _forward(self, densinfo):
        return self.a * safepow(densinfo.density.abs(), self.p)

    def _potential(self, densinfo):
        return self.p * self.a * safepow(densinfo.density.abs(), self.p - 1)

    def getfwdparamnames(self, prefix=""):
        return [prefix + "a", prefix + "p"]
