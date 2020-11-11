import torch
import numpy as np
from ddft.eks import BaseEKS, Hartree
from ddft.utils.safeops import safepow
from ddft.utils.datastruct import DensityInfo
from ddft.grids.base_grid import BaseRadialAngularGrid, Base3DGrid

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

def run_hartree_test(gridname, fcnname, rtol=1e-5, atol=1e-8):
    dtype = torch.float64
    grid, density = _setup_density(gridname, fcnname, dtype=dtype)
    half_density = density * 0.5
    half_densinfo = DensityInfo(density=half_density)

    hartree_mdl = Hartree()
    hartree_mdl.set_grid(grid)
    eks_hartree = hartree_mdl.forward(half_densinfo, half_densinfo)

    def eks_sum(density):
        half_densinfo = DensityInfo(density = 0.5 * density)
        eks_grid = hartree_mdl(half_densinfo, half_densinfo)
        return eks_grid.sum()

    vks_poisson = grid.solve_poisson(-4.0 * np.pi * density)
    eks_poisson = vks_poisson * 0.5 * density
    assert torch.allclose(eks_hartree, eks_poisson, rtol=rtol, atol=atol)

def _setup_density(gridname, fcnname, dtype=torch.float64):
    from ddft.grids.radialgrid import LegendreShiftExpRadGrid
    from ddft.grids.sphangulargrid import Lebedev
    from ddft.grids.multiatomsgrid import BeckeMultiGrid

    if gridname == "legradialshiftexp":
        grid = LegendreShiftExpRadGrid(200, 1e-6, 1e4, dtype=dtype)
    elif gridname == "lebedev":
        radgrid = LegendreShiftExpRadGrid(200, 1e-6, 1e4, dtype=dtype)
        grid = Lebedev(radgrid, prec=13, basis_maxangmom=3, dtype=dtype)
    elif gridname == "becke":
        atompos = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) # (natom, ndim)
        radgrid = LegendreShiftExpRadGrid(200, 1e-6, 1e2, dtype=dtype)
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
