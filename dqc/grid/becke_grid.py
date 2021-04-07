import torch
import numpy as np
from typing import List, Optional, Tuple
from dqc.grid.base_grid import BaseGrid
from dqc.grid.lebedev_grid import LebedevGrid

class BeckeGrid(BaseGrid):
    """
    Using Becke's scheme to construct the 3D grid consists of multiple 3D grids
    centered on each atom
    """

    def __init__(self, atomgrid: List[LebedevGrid], atompos: torch.Tensor) -> None:
        # atomgrid: list with length (natoms)
        # atompos: (natoms, ndim)

        assert atompos.shape[0] == len(atomgrid), \
            "The lengths of atomgrid and atompos must be the same"
        assert len(atomgrid) > 0
        self._dtype = atomgrid[0].dtype
        self._device = atomgrid[0].device

        # construct the grid points positions, weights, and the index of grid corresponding to each atom
        rgrids, self._rgrid, dvol_atoms = _construct_rgrids(atomgrid, atompos)

        # calculate the integration weights
        weights_atoms = _get_atom_weights(rgrids, atompos)  # (ngrid,)
        self._dvolume = dvol_atoms * weights_atoms

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def coord_type(self):
        return "cart"

    def get_dvolume(self):
        return self._dvolume

    def get_rgrid(self) -> torch.Tensor:
        return self._rgrid

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_rgrid":
            return [prefix + "_rgrid"]
        elif methodname == "get_dvolume":
            return [prefix + "_dvolume"]
        else:
            raise KeyError("Invalid methodname: %s" % methodname)


def _construct_rgrids(atomgrid: List[LebedevGrid], atompos: torch.Tensor) \
        -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    # construct the grid positions in a 2D tensor, the weights per isolated atom

    allpos_lst = [
        # rgrid: (ngrid[i], ndim), pos: (ndim,)
        (gr.get_rgrid() + pos) \
        for (gr, pos) in zip(atomgrid, atompos)]
    rgrid = torch.cat(allpos_lst, dim=0)  # (ngrid, ndim)

    # calculate the dvol for an isolated atom
    dvol_atoms = torch.cat([gr.get_dvolume() for gr in atomgrid], dim=0)

    return allpos_lst, rgrid, dvol_atoms

def _get_atom_weights(rgrids: List[torch.Tensor], atompos: torch.Tensor,
                      atomradius: Optional[torch.Tensor] = None) -> torch.Tensor:
    # rgrids: list of (natgrid, ndim) with length natoms consisting of absolute position of the grids
    # atompos: (natoms, ndim)
    # atomradius: (natoms,) or None
    # returns: (ngrid,)
    assert len(rgrids) == atompos.shape[0]

    natoms = atompos.shape[0]
    rdatoms = atompos - atompos.unsqueeze(1)  # (natoms, natoms, ndim)
    # add the diagonal to stabilize the gradient calculation
    rdatoms = rdatoms + torch.eye(rdatoms.shape[0], dtype=rdatoms.dtype,
                                  device=rdatoms.device).unsqueeze(-1)
    ratoms = torch.norm(rdatoms, dim=-1)  # (natoms, natoms)

    # calculate the distortion due to heterogeneity
    # (Appendix in Becke's https://doi.org/10.1063/1.454033)
    if atomradius is not None:
        chiij = atomradius / atomradius.unsqueeze(1)  # (natoms, natoms)
        uij = (atomradius - atomradius.unsqueeze(1)) / \
              (atomradius + atomradius.unsqueeze(1))
        aij = torch.clamp(uij / (uij * uij - 1), min=-0.45, max=0.45)  # (natoms, natoms)
        aij = aij.unsqueeze(-1)  # (natoms, natoms, 1)

    w_list: List[torch.Tensor] = []
    for ia, xyz in enumerate(rgrids):
        # xyz: (natgrid, ndim)
        rgatoms = torch.norm(xyz - atompos.unsqueeze(1), dim=-1)  # (natoms, natgrid)
        mu_ij = (rgatoms - rgatoms.unsqueeze(1)) / ratoms.unsqueeze(-1)  # (natoms, natoms, natgrid)

        if atomradius is not None:
            mu_ij = mu_ij + aij * (1 - mu_ij * mu_ij)

        f = mu_ij
        for _ in range(3):
            f = 0.5 * f * (3 - f * f)

        # small epsilon to avoid nan in the gradient
        s = 0.5 * (1. + 1e-12 - f)  # (natoms, natoms, natgrid)
        s = s + 0.5 * torch.eye(natoms).unsqueeze(-1)
        p = s.prod(dim=0)  # (natoms, natgrid)
        p = p / p.sum(dim=0, keepdim=True)  # (natoms, natgrid)

        w_list.append(p[ia])

    w = torch.cat(w_list, dim=-1)  # (ngrid)
    return w
