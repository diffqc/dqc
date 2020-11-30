import os
from typing import List
import torch
import numpy as np
from dqc.grid.base_grid import BaseGrid
from dqc.grid.radial_grid import RadialGrid

class LebedevGrid(BaseGrid):
    """
    Using Lebedev predefined angular points + radial grid to form 3D grid.
    """

    def __init__(self, radgrid: RadialGrid, prec: int) -> None:
        self._dtype = radgrid.dtype
        self._device = radgrid.device

        assert (prec % 2 == 1) and (3 <= prec <= 131),\
            "Precision must be an odd number between 3 and 131"

        # load the Lebedev grid points
        dset_path = os.path.join(os.path.split(__file__)[0], "..", "datasets",
                                 "lebedevquad", "lebedev_%03d.txt" % prec)
        assert os.path.exists(dset_path), "The dataset lebedev_%03d.txt does not exist" % prec
        lebedev_dsets = torch.tensor(np.loadtxt(dset_path),
                                     dtype=self._dtype, device=self._device)
        wphitheta = lebedev_dsets[:, -1]  # (nphitheta)
        phi = lebedev_dsets[:, 0] * (np.pi / 180.0)
        theta = lebedev_dsets[:, 1] * (np.pi / 180.0)

        # get the radial grid
        assert radgrid.coord_type == "radial"
        r = radgrid.get_rgrid().unsqueeze(-1)  # (nr, 1)

        # get the cartesian coordinate
        rsintheta = r * torch.sin(theta)
        x = (rsintheta * torch.cos(phi)).view(-1, 1)  # (nr * nphitheta, 1)
        y = (rsintheta * torch.sin(phi)).view(-1, 1)
        z = (r * torch.cos(theta)).view(-1, 1)
        xyz = torch.cat((x, y, z), dim=-1)  # (nr * nphitheta, ndim)
        self._xyz = xyz

        # calculate the dvolume (integration weights)
        dvol_rad = radgrid.get_dvolume().unsqueeze(-1)  # (nr, 1)
        self._dvolume = (dvol_rad * wphitheta).view(-1)  # (nr * nphitheta)

    def get_rgrid(self) -> torch.Tensor:
        return self._xyz

    def get_dvolume(self) -> torch.Tensor:
        return self._dvolume

    @property
    def coord_type(self) -> str:
        return "cart"

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_rgrid":
            return [prefix + "_xyz"]
        elif methodname == "get_dvolume":
            return [prefix + "_dvolume"]
        else:
            raise KeyError("Invalid methodname: %s" % methodname)
