import torch
import lintorch as lt

__all__ = ["VKS"]

class VKS(torch.nn.Module, lt.EditableModule):
    def __init__(self, eks_model, grid, use_potential=True):
        super(VKS, self).__init__()
        self.eks_model = eks_model
        self.eks_model.set_grid(grid)
        self.use_potential = use_potential
        self.grid = grid
        self.dv = self.grid.get_dvolume()

    def forward(self, x):
        assert x.ndim == 2, "The input to VKS module must be 2-dimensional tensor (nbatch, nrgrid)"
        if self.use_potential:
            return self.eks_model.potential(x)

        if x.requires_grad:
            xinp = x
        else:
            xinp = x.clone().requires_grad_()

        with torch.enable_grad():
            y = self.eks_model(xinp) # (nbatch,nr)
            y = y * self.dv
            ysum = y.sum()
        grad_enabled = torch.is_grad_enabled()
        dx = torch.autograd.grad(ysum, (xinp,),
            create_graph=grad_enabled)[0]
        return dx / self.dv

    def getparams(self, methodname):
        if methodname == "forward" or methodname == "__call__":
            if self.use_potential:
                return self.eks_model.getparams("potential")
            else:
                return [self.dv] + self.eks_model.getparams("forward")
        else:
            raise RuntimeError("The method %s has not been specified for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "forward" or methodname == "__call__":
            if self.use_potential:
                self.eks_model.setparams("potential", *params)
            else:
                self.dv = params[0]
                self.eks_model.setparams("forward", *params[1:])
        else:
            raise RuntimeError("The method %s has not been specified for setparams" % methodname)

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
    from ddft.grids.radialgrid import LegendreRadialShiftExp

    dtype = torch.float64
    grid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype)
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
