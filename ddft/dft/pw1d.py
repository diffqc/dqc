import torch
from ddft.modules.base_linear import BaseLinearModule
from ddft.modules.eigen import EigenModule
from ddft.modules.equilibrium import EquilibriumModule
from ddft.utils.misc import set_default_option

class HamiltonPW1D(BaseLinearModule):
    def __init__(self, boxsize, max_energy):
        super(HamiltonPW1D, self).__init__()
        self.boxsize = boxsize
        self.max_energy = max_energy
        pass

    def forward(self, wf, vext):
        # wf: (nbatch, nr)
        # vext: (nbatch, nr)
        pass

    @property
    def shape(self):
        pass

    def diag(self, vext):
        pass

class VKS1(torch.nn.Module):
    def __init__(self, a, p):
        super(VKS1, self).__init__()
        self.a = torch.nn.Parameter(a)
        self.p = torch.nn.Parameter(p)

    def forward(self, density):
        vks = self.a * density.abs()**self.p
        return vks

class DFTPW1D(torch.nn.Module):
    def __init__(self):
        super(DFTPW1D, self).__init__()
        pass

    def forward(self, density, vext, focc):
        # density: (nbatch, nr)
        # vext: (nbatch, nr)
        # focc: (nbatch, nlowest)
        pass

if __name__ == "__main__":
    pass
