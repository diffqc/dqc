import torch
import numpy as np
from ddft.eks import BaseEKS, VKS, Hartree
from ddft.utils.safeops import safepow

class EKS1(BaseEKS):
    def __init__(self, a, p):
        super(EKS1, self).__init__()
        self.a = torch.nn.Parameter(a)
        self.p = torch.nn.Parameter(p)

    def forward(self, x):
        return self.a * x**self.p

def test_vks():
    dtype = torch.float64
    grid, density = _setup_density("radialshiftexp", "exp", dtype=dtype)

    a = torch.tensor([1.0]).to(dtype)
    p = torch.tensor([1.3333]).to(dtype)
    eks_mdl = EKS1(a, p)
    vks_mdl = VKS(eks_mdl, grid)
    eks = eks_mdl(density)
    vks = vks_mdl(density)

    torch.allclose(eks, a*density**p)
    torch.allclose(vks, a*p*density**(p-1.0))

def test_hartree_radial():
    dtype = torch.float64
    grid, density = _setup_density("radialshiftexp", "exp", dtype=dtype)

    hartree_mdl = Hartree(grid)
    vks_hartree_mdl = VKS(hartree_mdl, grid)
    vks_hartree = vks_hartree_mdl(density)

    def eks_sum(density):
        eks_grid = hartree_mdl(density)
        return eks_grid.sum()

    vks_poisson = grid.solve_poisson(-4.0 * np.pi * density)
    assert torch.allclose(vks_hartree, vks_poisson)

def _setup_density(gridname, fcnname, dtype=torch.float64):
    from ddft.grids.radialshiftexp import RadialShiftExp

    if gridname == "radialshiftexp":
        grid = RadialShiftExp(1e-6, 1e4, 2000, dtype=dtype)
    else:
        raise RuntimeError("Unknown gridname: %s" % gridname)
    rgrid = grid.rgrid
    if fcnname == "exp":
        density = torch.exp(-rgrid[:,0]).unsqueeze(0)
    else:
        raise RuntimeError("Unknown fcnname: %s" % fcnname)
    return grid, density
