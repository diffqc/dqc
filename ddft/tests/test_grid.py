from itertools import product
import torch
import numpy as np
from ddft.grids.radialshiftexp import RadialShiftExp, LegendreRadialShiftExp
from ddft.grids.sphangulargrid import Lebedev

radial_gridnames = ["radialshiftexp", "legradialshiftexp"]
radial_fcnnames = ["gauss1", "exp1"]
sph_gridnames = ["lebedev"]
sph_fcnnames = ["radial_gauss1", "radial_exp1"]

def test_integralbox():
    # test if the integral (basis^2) dVolume in the given grid should be
    # equal to 1 where the basis is a pre-computed normalized basis function

    dtype = torch.float64
    device = torch.device("cpu")

    def runtest(gridname, fcnname):
        grid, rtol = get_grid(gridname, dtype, device)

        # get some profiles
        ones = torch.tensor([1.0], dtype=dtype, device=device)
        prof1 = get_fcn(fcnname, grid.rgrid) # (nr, nbasis)
        int1 = grid.integralbox(prof1*prof1, dim=0)
        assert torch.allclose(int1, ones, rtol=rtol, atol=0.0)

    for gridname, fcnname in product(radial_gridnames, radial_fcnnames):
        runtest(gridname, fcnname)
    for gridname, fcnname in product(sph_gridnames, sph_fcnnames):
        runtest(gridname, fcnname)

def get_grid(gridname, dtype, device):
    if gridname == "radialshiftexp":
        grid = RadialShiftExp(1e-6, 1e4, 2000, dtype=dtype, device=device)
        rtol = 1e-4
    elif gridname == "legradialshiftexp":
        grid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype, device=device)
        rtol = 1e-8
    elif gridname == "lebedev":
        radgrid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype, device=device)
        grid = Lebedev(radgrid, prec=13, basis_maxangmom=3, dtype=dtype, device=device)
        rtol = 1e-8
    return grid, rtol

def get_fcn(fcnname, rgrid):
    if fcnname in radial_fcnnames:
        return get_radial_fcn(fcnname, rgrid)
    elif fcnname in sph_fcnnames:
        if fcnname.startswith("radial_"):
            return get_radial_fcn(fcnname[7:], rgrid)

    raise RuntimeError("Unknown function name: %s" % fcnname)

def get_radial_fcn(fcnname, rgrid):
    rs = rgrid[:,0].unsqueeze(-1) # (nr,1)
    dtype = rgrid.dtype
    device = rgrid.device
    gw = torch.logspace(np.log10(1e-4), np.log10(1e2), 100).to(dtype).to(device)
    if fcnname == "gauss1":
        unnorm_basis = torch.exp(-rs*rs / (2*gw*gw)) * rs # (nr,ng)
        norm = np.sqrt(2./3) / gw**2.5 / np.pi**.75 # (ng)
        return unnorm_basis * norm # (nr, ng)
    elif fcnname == "exp1":
        unnorm_basis = torch.exp(-rs/gw)
        norm = 1./torch.sqrt(np.pi*gw**3)
        return unnorm_basis * norm
    else:
        raise RuntimeError("Unknown function name: %s" % fcnname)
