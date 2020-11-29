from abc import abstractmethod
import torch
import numpy as np
from dqc.grid.base_grid import BaseGrid
from typing import Tuple

__all__ = ["RadialGrid"]

class RadialGrid(BaseGrid):
    """
    Grid for radially symmetric system. This grid consists of two specifiers:
    * grid_integrator, and
    * grid_transform

    grid_integrator is to specify how to perform an integration on a fixed
    interval from -1 to 1.

    grid_transform is to transform the integration from the coordinate of
    grid_integrator to the actual coordinate.
    """
    def __init__(self, ngrid: int, grid_integrator: str = "chebyshev",
                 grid_transform: str = "logm3",
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu')):
        grid_transform_obj  = get_grid_transform(grid_transform)

        # get the location and weights of the integration in its original
        # coordinate
        x, w = get_xw_integration(ngrid, grid_integrator)
        x = torch.as_tensor(x, dtype=dtype, device=device)
        w = torch.as_tensor(w, dtype=dtype, device=device)
        r = grid_transform_obj.x2r(x)  # (ngrid,)

        # get the coordinate in Cartesian
        r1 = r.unsqueeze(-1)  # (ngrid, 1)
        self.rgrid = r1
        # r1_zeros = torch.zeros_like(r1)
        # self.rgrid = torch.cat((r1, r1_zeros, r1_zeros), dim = -1)

        # integration element
        drdx = grid_transform_obj.get_drdx(r)
        vol_elmt = 4 * np.pi * r * r  # (ngrid,)
        dr = drdx * w
        self.dvolume = vol_elmt * dr  # (ngrid,)

    def get_dvolume(self) -> torch.Tensor:
        return self.dvolume

    def get_rgrid(self) -> torch.Tensor:
        return self.rgrid

    def getparamnames(self, methodname: str, prefix: str = ""):
        if methodname == "get_dvolume":
            return [prefix + "dvolume"]
        elif methodname == "get_rgrid":
            return [prefix + "rgrid"]
        else:
            raise KeyError("getparamnames for %s is not set" % methodname)

def get_xw_integration(n: int, s0: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns ``n`` points of integration from -1 to 1 and its integration
    # weights

    s = s0.lower()
    if s == "chebyshev":
        # generate the x and w from chebyshev polynomial
        np1 = n + 1.
        icount = np.arange(n, 0, -1)
        ipn1 = icount * np.pi / np1
        sin_ipn1 = np.sin(ipn1)
        sin_ipn1_2 = sin_ipn1 * sin_ipn1
        xcheb = (np1 - 2 * icount) / np1 + 2 / np.pi * \
                (1 + 2. / 3 * sin_ipn1 * sin_ipn1) * np.cos(ipn1) * sin_ipn1
        wcheb = 16. / (3 * np1) * sin_ipn1_2 * sin_ipn1_2
        return xcheb, wcheb
    else:
        raise RuntimeError("Unknown grid_integrator: %s" % s0)

### grid transformation ###

class BaseGridTransform(object):
    @abstractmethod
    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        # transform from x (coordinate from -1 to 1) to r coordinate (0 to inf)
        pass

    @abstractmethod
    def get_drdx(self, r: torch.Tensor) -> torch.Tensor:
        # returns the dr/dx
        pass

class LogM3Transformation(BaseGridTransform):
    # eq (12) in https://aip.scitation.org/doi/pdf/10.1063/1.475719
    def __init__(self, ra: float = 1.0, eps: float = 1e-15):
        self.ra = ra
        self.eps = eps
        self.ln2 = np.log(2.0 + eps)

    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        return self.ra * (1 - torch.log1p(-x + self.eps) / self.ln2)

    def get_drdx(self, r: torch.Tensor) -> torch.Tensor:
        return self.ra / self.ln2 * torch.exp(-self.ln2 * (1.0 - r / self.ra))

def get_grid_transform(s0: str) -> BaseGridTransform:
    s = s0.lower()
    if s == "logm3":
        return LogM3Transformation()
    else:
        raise RuntimeError("Unknown grid transformation: %s" % s0)
