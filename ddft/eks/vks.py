import torch
import xitorch as xt

__all__ = ["VKS"]

class VKS(torch.nn.Module, xt.EditableModule):
    def __init__(self, eks_model, grid):
        super(VKS, self).__init__()
        self.eks_model = eks_model
        self.eks_model.set_grid(grid)
        self.grid = grid

    def forward(self, n, gradn=None):
        assert n.ndim == 2, "The input to VKS module must be 2-dimensional tensor (nbatch, nrgrid)"
        return self.eks_model.potential(n)

    def getparamnames(self, methodname, prefix=""):
        if methodname == "forward" or methodname == "__call__":
            return self.eks_model.getparamnames("potential", prefix=prefix+"eks_model.")
        else:
            raise KeyError("Getparamnames has no %s method" % methodname)


if __name__ == "__main__":
    from ddft.eks.base_eks import BaseEKS
    from ddft.utils.safeops import safepow

    class EKS1(BaseEKS):
        def __init__(self, a, p):
            super(EKS1, self).__init__()
            self.a = torch.nn.Parameter(a)
            self.p = torch.nn.Parameter(p)

        def forward(self, x):
            return self.a * safepow(x, self.p)

    import matplotlib.pyplot as plt
    from ddft.grids.radialgrid import LegendreShiftExpRadGrid

    dtype = torch.float64
    grid = LegendreShiftExpRadGrid(200, 1e-6, 1e4, dtype=dtype)
    rgrid = grid.rgrid
    density = torch.exp(-rgrid[:,0]).unsqueeze(0)

    a = torch.tensor([1.0]).to(dtype)
    p = torch.tensor([1.3333]).to(dtype)
    eks_mdl = EKS1(a, p)
    vks_mdl = VKS(eks_mdl, grid)
    eks = eks_mdl(density)
    vks = vks_mdl(density)

    tonp = lambda x: x.detach().numpy().ravel()
    plt.plot(tonp(vks))
    plt.plot(tonp(a * p * density ** (p-1.0)))
    plt.show()
