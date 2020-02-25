from itertools import product
import torch
import numpy as np
from ddft.grids.radialshiftexp import RadialShiftExp, LegendreRadialShiftExp
from ddft.grids.sphangulargrid import Lebedev

radial_gridnames = ["radialshiftexp", "legradialshiftexp"]
radial_fcnnames = ["gauss1", "exp1"]
sph_gridnames = ["lebedev"]

dtype = torch.float64
device = torch.device("cpu")

def test_radial_integralbox():
    # test if the integral (basis^2) dVolume in the given grid should be
    # equal to 1 where the basis is a pre-computed normalized basis function
    def runtest(gridname, fcnname):
        grid, rtol = get_radial_grid(gridname, dtype, device)
        prof1 = get_fcn(fcnname, grid.rgrid) # (nr, nbasis)
        runtest_integralbox(grid, rtol, prof1)

    for gridname, fcnname in product(radial_gridnames, radial_fcnnames):
        runtest(gridname, fcnname)

def test_spherical_integralbox():
    def runtest(spgridname, radgridname, fcnname):
        radgrid, rtol = get_radial_grid(radgridname, dtype, device)
        sphgrid = get_spherical_grid(spgridname, radgrid, dtype, device)
        prof1 = get_fcn(fcnname, sphgrid.rgrid)
        runtest_integralbox(sphgrid, rtol, prof1)

    for gridname, radgridname, fcnname in product(sph_gridnames, radial_gridnames, radial_fcnnames):
        runtest(gridname, radgridname, fcnname)

# def test_radial_poisson():
#     def runtest(gridname, fcnname):
#         grid, rtol = get_radial_grid(gridname, dtype, device)
#         prof1 = get_fcn(fcnname, grid.rgrid)
#         poisson1 = get_poisson(fcnname, grid.rgrid)
#         runtest_poisson(grid, rtol, prof1, poisson1)
#
#     for gridname, fcnname in product(radial_gridnames, radial_fcnnames):
#         runtest(gridname, fcnname)

############################## helper functions ##############################
def runtest_integralbox(grid, rtol, prof):
    ones = torch.tensor([1.0], dtype=prof.dtype, device=prof.device)
    int1 = grid.integralbox(prof*prof, dim=0)
    assert torch.allclose(int1, ones, rtol=rtol, atol=0.0)

def runtest_poisson(grid, rtol, prof, poisson):
    pois = grid.solve_poisson(prof.transpose(-2,-1)).transpose(-2,-1)
    assert torch.allclose(pois, poisson, rtol=rtol)

def get_radial_grid(gridname, dtype, device):
    if gridname == "radialshiftexp":
        grid = RadialShiftExp(1e-6, 1e4, 2000, dtype=dtype, device=device)
        rtol = 1e-4
    elif gridname == "legradialshiftexp":
        grid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype, device=device)
        rtol = 1e-8
    else:
        raise RuntimeError("Unknown radial grid name: %s" % gridname)
    return grid, rtol

def get_spherical_grid(gridname, radgrid, dtype, device):
    if gridname == "lebedev":
        grid = Lebedev(radgrid, prec=13, basis_maxangmom=3, dtype=dtype, device=device)
    else:
        raise RuntimeError("Unknown spherical grid name: %s" % gridname)
    return grid

def get_fcn(fcnname, rgrid):
    dtype = rgrid.dtype
    device = rgrid.device

    if fcnname in ["gauss1", "exp1"]:
        rs = rgrid[:,0].unsqueeze(-1) # (nr,1)
        gw = torch.logspace(np.log10(1e-4), np.log10(1e2), 100).to(dtype).to(device)
        if fcnname == "gauss1":
            unnorm_basis = torch.exp(-rs*rs / (2*gw*gw)) * rs # (nr,ng)
            norm = np.sqrt(2./3) / gw**2.5 / np.pi**.75 # (ng)
            return unnorm_basis * norm # (nr, ng)
        elif fcnname == "exp1":
            unnorm_basis = torch.exp(-rs/gw)
            norm = 1./torch.sqrt(np.pi*gw**3)
            return unnorm_basis * norm

    raise RuntimeError("Unknown function name: %s" % fcnname)

def get_poisson(fcnname, rgrid):
    # ???
    dtype = rgrid.dtype
    device = rgrid.device

    if fcnname in ["gauss1", "exp1"]:
        rs = rgrid[:,0].unsqueeze(-1) # (nr,1)
        gw = torch.logspace(np.log10(1e-4), np.log10(1e2), 100).to(dtype).to(device)
        if fcnname == "gauss1":
            unnorm_basis = torch.exp(-rs*rs / (2*gw*gw)) * rs # (nr,ng)
            norm = np.sqrt(2./3) / gw**2.5 / np.pi**.75 # (ng)
            return unnorm_basis * norm # (nr, ng)
        elif fcnname == "exp1":
            unnorm_basis = torch.exp(-rs/gw)
            norm = 1./torch.sqrt(np.pi*gw**3)
            return unnorm_basis * norm
