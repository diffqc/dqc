import torch
import numpy as np
import pytest
from dqc.grid.radial_grid import RadialGrid
from dqc.grid.lebedev_grid import LebedevGrid

rgrid_combinations = [
    ("chebyshev", "logm3"),
]

@pytest.mark.parametrize(
    "grid_integrator,grid_transform",
    rgrid_combinations
)
def test_radial_grid_dvol(grid_integrator, grid_transform):
    ngrid = 40
    dtype = torch.float64
    radgrid = RadialGrid(ngrid, grid_integrator, grid_transform, dtype=dtype)

    dvol = radgrid.get_dvolume()  # (ngrid,)
    rgrid = radgrid.get_rgrid()  # (ngrid, ndim)
    r = rgrid[:, 0]

    # test gaussian integration
    fcn = torch.exp(-r * r * 0.5)  # (ngrid,)
    int1 = (fcn * dvol).sum()
    val1 = 2 * np.sqrt(2 * np.pi) * np.pi
    assert torch.allclose(int1, int1 * 0 + val1)

@pytest.mark.parametrize(
    "rgrid_integrator,rgrid_transform",
    rgrid_combinations
)
def test_lebedev_grid_dvol(rgrid_integrator, rgrid_transform):
    dtype = torch.float64
    nr = 40
    prec = 7
    radgrid = RadialGrid(nr, rgrid_integrator, rgrid_transform, dtype=dtype)
    sphgrid = LebedevGrid(radgrid, prec=prec)

    dvol = sphgrid.get_dvolume()  # (ngrid,)
    rgrid = sphgrid.get_rgrid()  # (ngrid, ndim)
    x = rgrid[:, 0]
    y = rgrid[:, 1]
    z = rgrid[:, 2]

    # test gaussian integration
    fcn = torch.exp(-(x * x + y * y + z * z) * 0.5)
    int1 = (fcn * dvol).sum()
    val1 = 2 * np.sqrt(2 * np.pi) * np.pi
    assert torch.allclose(int1, int1 * 0 + val1)
