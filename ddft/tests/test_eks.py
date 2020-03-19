import torch
import numpy as np
from ddft.eks import BaseEKS, VKS, Hartree
from ddft.utils.safeops import safepow
from ddft.grids.base_grid import BaseRadialAngularGrid, Base3DGrid

class EKS1(BaseEKS):
    def __init__(self, a, p):
        super(EKS1, self).__init__()
        self.a = torch.nn.Parameter(a)
        self.p = torch.nn.Parameter(p)

    def forward(self, x):
        return self.a * x**self.p

def test_vks_radial_legendre():
    run_vks_test("legradialshiftexp", "exp")

def test_vks_lebedev():
    run_vks_test("lebedev", "gauss-l1")
    run_vks_test("lebedev", "gauss-l2")
    run_vks_test("lebedev", "gauss-l1m1")
    run_vks_test("lebedev", "gauss-l2m2")

def test_vks_becke():
    run_vks_test("becke", "exp")
    run_vks_test("becke", "exp-twocentres")

def test_hartree_radial_legendre():
    run_hartree_test("legradialshiftexp", "exp")

def test_hartree_lebedev():
    run_hartree_test("lebedev", "gauss-l1")
    run_hartree_test("lebedev", "gauss-l2")
    run_hartree_test("lebedev", "gauss-l1m1")
    run_hartree_test("lebedev", "gauss-l2m2")

def test_hartree_becke():
    rtol, atol = 6e-3, 1e-2
    run_hartree_test("becke", "exp", rtol=rtol, atol=atol)
    run_hartree_test("becke", "exp-twocentres", rtol=rtol, atol=atol)

def run_vks_test(gridname, fcnname, rtol=1e-5, atol=1e-8):
    dtype = torch.float64
    grid, density = _setup_density(gridname, fcnname, dtype=dtype)

    a = torch.tensor([1.0]).to(dtype)
    p = torch.tensor([1.3333]).to(dtype)
    eks_mdl = EKS1(a, p)
    # use_potential=False to use the derivative to check if the poisson
    # solver is stable and implemented correctly
    vks_mdl = VKS(eks_mdl, grid, use_potential=False)
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
    from ddft.grids.radialgrid import LegendreRadialShiftExp
    from ddft.grids.sphangulargrid import Lebedev
    from ddft.grids.multiatomsgrid import BeckeMultiGrid

    if gridname == "legradialshiftexp":
        grid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype)
    elif gridname == "lebedev":
        radgrid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype)
        grid = Lebedev(radgrid, prec=13, basis_maxangmom=3, dtype=dtype)
    elif gridname == "becke":
        atompos = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) # (natom, ndim)
        radgrid = LegendreRadialShiftExp(1e-6, 1e2, 200, dtype=dtype)
        sphgrid = Lebedev(radgrid, prec=13, basis_maxangmom=8, dtype=dtype)
        grid = BeckeMultiGrid(sphgrid, atompos, dtype=dtype)
    else:
        raise RuntimeError("Unknown gridname: %s" % gridname)

    rgrid = grid.rgrid # (nr, ndim)
    if rgrid.shape[1] == 1:
        rs = rgrid[:,0]
    if rgrid.shape[1] == 3:
        if isinstance(grid, Base3DGrid):
            xyzgrid = grid.rgrid_in_xyz # (nr, ndim)
        if isinstance(grid, BaseRadialAngularGrid):
            rs = rgrid[:,0]
            phi = rgrid[:,1]
            theta = rgrid[:,2]
        else:
            x = rgrid[:,0]
            y = rgrid[:,1]
            z = rgrid[:,2]
            rs = rgrid.norm(dim=-1) # (nr,)
            xy = rgrid[:,:2].norm(dim=-1)
            phi = torch.atan2(y, x)
            theta = torch.atan2(xy, z)

    if fcnname == "exp":
        density = torch.exp(-rs)
    elif fcnname == "gauss-l1":
        density = torch.exp(-rs*rs/2) * torch.cos(theta)
    elif fcnname == "gauss-l2":
        density = torch.exp(-rs*rs/2) * (3*torch.cos(theta)**2-1)/2.0
    elif fcnname == "gauss-l1m1":
        density = torch.exp(-rs*rs/2) * torch.sin(theta) * torch.cos(phi)
    elif fcnname == "gauss-l2m2":
        density = torch.exp(-rs*rs/2) * 3*torch.sin(theta)**2 * torch.cos(2*phi) # (nr,1)
    elif fcnname == "exp-twocentres":
        dist = (xyzgrid - atompos.unsqueeze(1)).norm(dim=-1) # (natom, nr)
        density = torch.exp(-dist).sum(dim=0)
    else:
        raise RuntimeError("Unknown fcnname: %s" % fcnname)

    density = density.unsqueeze(0)
    return grid, density
