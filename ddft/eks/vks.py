import torch

__all__ = ["VKS"]

class VKS(torch.nn.Module):
    def __init__(self, eks_model):
        super(VKS, self).__init__()
        self.eks_model = eks_model

    def forward(self, x):
        assert x.ndim == 2, "The input to VKS module must be 2-dimensional tensor (nbatch, nrgrid)"
        if x.requires_grad:
            xinp = x
        else:
            xinp = x.clone().requires_grad_()

        with torch.enable_grad():
            y = self.eks_model(xinp) # (nbatch,nr)
            ysum = y.sum()
        grad_enabled = torch.is_grad_enabled()
        dx = torch.autograd.grad(ysum, (xinp,),
            create_graph=grad_enabled)[0]
        return dx # (same shape as x)

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
    from ddft.grids.radialshiftexp import RadialShiftExp

    dtype = torch.float64
    grid = RadialShiftExp(1e-6, 1e4, 2000, dtype=dtype)
    rgrid = grid.rgrid
    density = torch.exp(-rgrid[:,0]).unsqueeze(0)

    a = torch.tensor([1.0]).to(dtype)
    p = torch.tensor([1.3333]).to(dtype)
    eks_mdl = EKS1(a, p)
    vks_mdl = VKS(eks_mdl)
    eks = eks_mdl(density)
    vks = vks_mdl(density)

    tonp = lambda x: x.detach().numpy().ravel()
    plt.plot(tonp(vks))
    plt.plot(tonp(a * p * density ** (p-1.0)))
    plt.show()
