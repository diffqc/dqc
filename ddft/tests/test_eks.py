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

def test_vks_radial():
    run_vks_test("radialshiftexp", "exp")

def test_vks_radial_legendre():
    run_vks_test("legradialshiftexp", "exp")

def test_vks_lebedev():
    run_vks_test("lebedev", "gauss-l1")
    run_vks_test("lebedev", "gauss-l2")

def test_hartree_radial():
    run_hartree_test("radialshiftexp", "exp")

def test_hartree_radial_legendre():
    run_hartree_test("legradialshiftexp", "exp")

def test_hartree_lebedev():
    run_hartree_test("lebedev", "gauss-l1", rtol=1e-3, atol=1e-3)
    run_hartree_test("lebedev", "gauss-l2", rtol=1e-3, atol=1e-3)

def run_vks_test(gridname, fcnname, rtol=1e-5, atol=1e-8):
    dtype = torch.float64
    grid, density = _setup_density(gridname, fcnname, dtype=dtype)

    a = torch.tensor([1.0]).to(dtype)
    p = torch.tensor([1.3333]).to(dtype)
    eks_mdl = EKS1(a, p)
    vks_mdl = VKS(eks_mdl, grid)
    eks = eks_mdl(density)
    vks = vks_mdl(density)

    torch.allclose(eks, a*density**p, rtol=rtol, atol=atol)
    torch.allclose(vks, a*p*density**(p-1.0), rtol=rtol, atol=atol)

def run_hartree_test(gridname, fcnname, rtol=1e-5, atol=1e-8):
    dtype = torch.float64
    grid, density = _setup_density(gridname, fcnname, dtype=dtype)

    hartree_mdl = Hartree(grid)
    vks_hartree_mdl = VKS(hartree_mdl, grid)
    vks_hartree = vks_hartree_mdl(density)

    def eks_sum(density):
        eks_grid = hartree_mdl(density)
        return eks_grid.sum()

    vks_poisson = grid.solve_poisson(-4.0 * np.pi * density)
    assert torch.allclose(vks_hartree, vks_poisson, rtol=rtol, atol=atol)

def _setup_density(gridname, fcnname, dtype=torch.float64):
    from ddft.grids.radialshiftexp import RadialShiftExp, LegendreRadialShiftExp
    from ddft.grids.sphangulargrid import Lebedev

    if gridname == "radialshiftexp":
        grid = RadialShiftExp(1e-6, 1e4, 2000, dtype=dtype)
    elif gridname == "legradialshiftexp":
        grid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype)
    elif gridname == "lebedev":
        radgrid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype)
        grid = Lebedev(radgrid, prec=13, basis_maxangmom=3, dtype=dtype)
    else:
        raise RuntimeError("Unknown gridname: %s" % gridname)

    rgrid = grid.rgrid
    rs = rgrid[:,0]
    if fcnname == "exp":
        density = torch.exp(-rs)
    elif fcnname == "gauss-l1":
        theta = rgrid[:,2]
        density = torch.exp(-rs*rs/2) * torch.cos(theta)
    elif fcnname == "gauss-l2":
        theta = rgrid[:,2]
        density = torch.exp(-rs*rs/2) * (3*torch.cos(theta)**2-1)/2.0
    else:
        raise RuntimeError("Unknown fcnname: %s" % fcnname)

    density = density.unsqueeze(0)
    return grid, density
