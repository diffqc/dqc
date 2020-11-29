import torch
import numpy as np
import pytest
from dqc.grid.radial_grid import RadialGrid

@pytest.mark.parametrize(
    "grid_integrator,grid_transform",
    [("chebyshev", "logm3")]
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
