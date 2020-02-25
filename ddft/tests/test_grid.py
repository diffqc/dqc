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
        grid = get_radial_grid(gridname, dtype, device)
        prof1 = get_fcn(fcnname, grid.rgrid) # (nr, nbasis)
        rtol, atol = get_rtol_atol("integralbox", gridname)
        runtest_integralbox(grid, prof1, rtol=rtol, atol=atol)

    for gridname, fcnname in product(radial_gridnames, radial_fcnnames):
        runtest(gridname, fcnname)

def test_spherical_integralbox():
    def runtest(spgridname, radgridname, fcnname):
        radgrid = get_radial_grid(radgridname, dtype, device)
        sphgrid = get_spherical_grid(spgridname, radgrid, dtype, device)
        prof1 = get_fcn(fcnname, sphgrid.rgrid)
        rtol, atol = get_rtol_atol("integralbox", spgridname, radgridname)
        runtest_integralbox(sphgrid, prof1, rtol=rtol, atol=atol)

    for gridname, radgridname, fcnname in product(sph_gridnames, radial_gridnames, radial_fcnnames):
        runtest(gridname, radgridname, fcnname)

def test_radial_poisson():
    def runtest(gridname, fcnname):
        grid = get_radial_grid(gridname, dtype, device)
        prof1 = get_fcn(fcnname, grid.rgrid)
        poisson1 = get_poisson(fcnname, grid.rgrid)
        rtol, atol = get_rtol_atol("poisson", gridname)
        runtest_poisson(grid, prof1, poisson1, rtol=rtol, atol=atol)

    for gridname, fcnname in product(radial_gridnames, radial_fcnnames):
        runtest(gridname, fcnname)

############################## helper functions ##############################
def runtest_integralbox(grid, prof, rtol, atol):
    ones = torch.tensor([1.0], dtype=prof.dtype, device=prof.device)
    int1 = grid.integralbox(prof*prof, dim=0)
    assert torch.allclose(int1, ones, rtol=rtol, atol=atol)

def runtest_poisson(grid, prof, poisson, rtol, atol):
    pois = grid.solve_poisson(prof.transpose(-2,-1)).transpose(-2,-1)
    # boundary condition
    pois = pois - pois[-1:,:]
    poisson = poisson - poisson[-1:,:]
    # normalize the scale
    pois = pois / pois.abs().max(dim=0)[0]
    poisson = poisson / poisson.abs().max(dim=0)[0]
    assert torch.allclose(pois, poisson, rtol=rtol, atol=atol)

def get_rtol_atol(taskname, gridname1, gridname2=None):
    rtolatol = {
        "integralbox": {
            # this is compared to 1, so rtol has the same effect as atol
            "radialshiftexp": [1e-4, 0.0],
            "legradialshiftexp": [1e-8, 0.0],
            "lebedev": {
                "radialshiftexp": [1e-4, 0.0],
                "legradialshiftexp": [1e-8, 0.0],
            }
        },
        "poisson": {
            "radialshiftexp": [0.0, 6e-3],
            "legradialshiftexp": [0.0, 6e-3],
        }
    }
    if gridname2 is None:
        return rtolatol[taskname][gridname1]
    else:
        return rtolatol[taskname][gridname1][gridname2]

def get_radial_grid(gridname, dtype, device):
    if gridname == "radialshiftexp":
        grid = RadialShiftExp(1e-6, 1e4, 2000, dtype=dtype, device=device)
    elif gridname == "legradialshiftexp":
        grid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype, device=device)
    else:
        raise RuntimeError("Unknown radial grid name: %s" % gridname)
    return grid

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
        gw = torch.logspace(np.log10(1e-4), np.log10(1e0), 30).to(dtype).to(device)
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
    dtype = rgrid.dtype
    device = rgrid.device

    if fcnname in ["gauss1", "exp1"]:
        rs = rgrid[:,0].unsqueeze(-1) # (nr,1)
        gw = torch.logspace(np.log10(1e-4), np.log10(1e0), 30).to(dtype).to(device)
        if fcnname == "gauss1":
            rg = rs/(np.sqrt(2)*gw)
            sqrtpi = np.sqrt(np.pi)
            y = -torch.sqrt(gw) * ((2*np.sqrt(2)*(1-torch.exp(-rg*rg))*gw + rs*sqrtpi) - rs*sqrtpi*torch.erf(rg)) / (np.sqrt(3)*rs*np.pi**.75)
            return y # (nr, ng)
        elif fcnname == "exp1":
            y = -gw*gw*(2*gw - torch.exp(-rs/gw)*(rs+2*gw)) / (rs*np.sqrt(np.pi)*gw**1.5)
            return y
