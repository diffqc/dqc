import torch
import numpy as np
from ddft.eks.base_eks import BaseEKS

__all__ = ["Hartree"]

class Hartree(BaseEKS):
    def forward(self, density):
        # density: (nbatch, nr)
        vks = self.potential(density)
        # dirichlet boundary
        # vks = vks - vks[:,-1]
        eks = 0.5 * vks * density
        return eks

    def potential(self, density):
        vks = self.grid.solve_poisson(-4.0*np.pi*density)
        return vks

    def getfwdparamnames(self, prefix=""):
        return self.grid.getparamnames("solve_poisson", prefix=prefix+"grid.")

    def getparamnames(self, methodname, prefix=""):
        return self.getfwdparamnames(prefix=prefix)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ddft.eks.vks import VKS
    from ddft.grids.radialgrid import LegendreShiftExpRadGrid
    from ddft.utils.fd import finite_differences

    dtype = torch.float64
    grid = LegendreShiftExpRadGrid(200, 1e-6, 1e4, dtype=dtype)
    rgrid = grid.rgrid
    density = torch.exp(-rgrid*rgrid).transpose(-2,-1)

    hartree_mdl = Hartree()
    hartree_mdl.set_grid(grid)
    vks_hartree_mdl = VKS(hartree_mdl, grid)
    eks_hartree = hartree_mdl(density)
    vks_hartree = vks_hartree_mdl(density)

    def eks_sum(density):
        eks_grid = hartree_mdl(density)
        return eks_grid.sum()

    vks_poisson = grid.solve_poisson(-4.0 * np.pi * density)
    tonp = lambda x: x.detach().numpy().ravel()
    print(vks_hartree.shape)
    plt.plot(tonp(vks_hartree))
    plt.plot(tonp(vks_poisson))
    plt.show()
