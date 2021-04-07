import torch
import numpy as np
from typing import List, Optional
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
        self._rgrid, dvol_atoms, cs_ngrid = _construct_rgrid_and_index(atomgrid, atompos)

        # calculate the integration weights
        weights_atoms = _get_atom_weights(self._rgrid, atompos, cs_ngrid)  # (ngrid,)
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

def _construct_rgrid_and_index(atomgrid: List[LebedevGrid], atompos: torch.Tensor) \
        -> Tuple[torch.Tensor, np.ndarray]:
    # construct the grid positions in a 2D tensor, the weights per isolated atom, and
    # also the index position for each atom in the returned grid

    allpos_lst = [
        # rgrid: (ngrid[i], ndim), pos: (ndim,)
        (gr.get_rgrid() + pos) \
        for (gr, pos) in zip(atomgrid, atompos)]
    rgrid = torch.cat(allpos_lst, dim=0)  # (ngrid, ndim)

    # get the index of grid corresponding to each atom
    cs_ngrid = np.cumsum([0] + [gr.get_rgrid().shape[-2] for gr in atomgrid])

    # calculate the dvol for an isolated atom
    dvol_atoms = torch.cat([gr.get_dvolume() for gr in atomgrid], dim=0)

    return rgrid, dvol_atoms, cs_ngrid

def _get_atom_weights(xyz: torch.Tensor, atompos: torch.Tensor,
                      atom_grids_idxs: np.ndarray,
                      atomradius: Optional[torch.Tensor] = None) -> torch.Tensor:
    # returns the relative integration weights for each atoms

    # rgrid: (ngrid, ndim)
    # atompos: (natoms, ndim)
    # atomradius: (natoms,) or None
    # atom_grids_idxs: (natoms + 1,)
    # returns: (ngrid,)

    natoms = atompos.shape[0]
    rgatoms = torch.norm(xyz - atompos.unsqueeze(1), dim=-1)  # (natoms, ngrid)
    rdatoms = atompos - atompos.unsqueeze(1)  # (natoms, natoms, ndim)
    # add the diagonal to stabilize the gradient calculation
    rdatoms = rdatoms + torch.eye(rdatoms.shape[0], dtype=rdatoms.dtype,
                                  device=rdatoms.device).unsqueeze(-1)
    ratoms = torch.norm(rdatoms, dim=-1)  # (natoms, natoms)
    mu_ij = (rgatoms - rgatoms.unsqueeze(1)) / ratoms.unsqueeze(-1)  # (natoms, natoms, ngrid)

    # calculate the distortion due to heterogeneity
    # (Appendix in Becke's https://doi.org/10.1063/1.454033)
    if atomradius is not None:
        chiij = atomradius / atomradius.unsqueeze(1)  # (natoms, natoms)
        uij = (atomradius - atomradius.unsqueeze(1)) / \
              (atomradius + atomradius.unsqueeze(1))
        aij = torch.clamp(uij / (uij * uij - 1), min=-0.45, max=0.45)
        mu_ij = mu_ij + aij * (1 - mu_ij * mu_ij)

    f = mu_ij
    for _ in range(3):
        f = 0.5 * f * (3 - f * f)
    # small epsilon to avoid nan in the gradient
    s = 0.5 * (1. + 1e-12 - f)  # (natoms, natoms, ngrid)
    s = s + 0.5 * torch.eye(natoms).unsqueeze(-1)
    p = s.prod(dim=0)  # (natoms, ngrid)
    p = p / p.sum(dim=0, keepdim=True)  # (natoms, ngrid)

    # check if the atomic grids all have the same number of points
    ngrids = atom_grids_idxs[1:] - atom_grids_idxs[:-1]
    same_sizes = np.all(ngrids == ngrids[0])

    # construct the grids
    if same_sizes:
        watoms0 = p.view(natoms, natoms, -1)  # (natoms, natoms, ngrid)
        watoms = watoms0.diagonal(dim1=0, dim2=1).transpose(-2, -1).reshape(-1)  # (natoms * ngrid)
    else:
        watoms_list = []
        for i in range(natoms):
            watoms_list.append(p[i, atom_grids_idxs[i]:atom_grids_idxs[i + 1]])  # (ngrid)
        watoms = torch.cat(watoms_list, dim=0)  # (ngrid,)
    return watoms
