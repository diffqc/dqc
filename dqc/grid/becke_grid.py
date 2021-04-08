import torch
import numpy as np
from typing import List, Optional, Tuple
from dqc.grid.base_grid import BaseGrid
from dqc.grid.lebedev_grid import LebedevGrid
from dqc.hamilton.intor.lattice import Lattice

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

class PBCBeckeGrid(BaseGrid):
    """
    Use Becke's scheme to construct the 3D grid in a periodic cell. It is similar
    to non-pbc BeckeGrid, but in this case, only grid points inside the lattice
    are considered, and atoms corresponds to each grid points are involved in
    calculating the weights.
    """
    def __init__(self, atomgrid: List[LebedevGrid], atompos: torch.Tensor, lattice: Lattice):
        # atomgrid: list with length (natoms)
        # atompos: (natoms, ndim)

        assert atompos.shape[0] == len(atomgrid), \
            "The lengths of atomgrid and atompos must be the same"
        assert len(atomgrid) > 0
        self._dtype = atomgrid[0].dtype
        self._device = atomgrid[0].device

        # get the normalized coordinates
        a = lattice.lattice_vectors()  # (nlvec=ndim, ndim)
        b = lattice.recip_vectors() / (2 * np.pi)  # (ndim, ndim) just the inverse of lattice vector.T

        new_atompos_lst: List[torch.Tensor] = []
        new_rgrids: List[torch.Tensor] = []
        new_dvols: List[torch.Tensor] = []
        for ia, atomgr in enumerate(atomgrid):
            atpos = atompos[ia]
            rgrid = atomgr.get_rgrid() + atpos  # (natgrid, ndim)
            dvols = atomgr.get_dvolume()  # (natgrid)

            # ugrid is the normalized coordinate
            ugrid = torch.einsum("cd,gd->gc", b, rgrid)  # (natgrid, ndim)

            # get the shift required to make the grid point inside the lattice
            ns = -ugrid.floor().to(torch.int)  # (natgrid, ndim) # ratoms + ns @ a will be the new atompos

            # ns_unique: (nunique, ndim), ns_unique_idx: (natgrid,), ns_count: (nunique)
            ns_unique, ns_unique_idx, ns_count = torch.unique(
                ns, dim=0, return_inverse=True, return_counts=True)

            # ignoring the shifts with only not more than 8 points (following pyscf)
            significant_uniq_idx = ns_count > 8  # (nunique)
            significant_idx = significant_uniq_idx[ns_unique_idx]  # (natgrid,)
            ns_unique = ns_unique[significant_uniq_idx, :]  # (nunique2, ndim)
            ls_unique = torch.matmul(ns_unique.to(a.dtype), a)  # (nunique2, ndim)

            # flag the unaccepted points with -1
            flag = -1
            ns_unique_idx[~significant_idx] = flag  # (natgrid)

            # get the coordinate inside the lattice
            ls = torch.matmul(ns.to(a.dtype), a)
            rg = rgrid + ls  # (natgrid, ndim)

            # get the new atom pos
            new_atpos = atompos[ia] + ls_unique  # (nunique2, ndim)
            new_atompos_lst.append(new_atpos)

            # separate the grid points that corresponds to the different atoms
            for idx in torch.unique(ns_unique_idx):
                if idx == flag:
                    continue
                at_idx = ns_unique_idx == idx
                new_rgrids.append(rg[at_idx, :])  # list of (natgrid2, ndim)
                new_dvols.append(dvols[at_idx])  # list of (natgrid2,)

        self._rgrid = torch.cat(new_rgrids, dim=0)  # (ngrid, ndim)
        dvol_atoms = torch.cat(new_dvols, dim=0)  # (ngrid)
        new_atompos = torch.cat(new_atompos_lst, dim=0)  # (nnewatoms, ndim)
        watoms = _get_atom_weights(new_rgrids, new_atompos)  # (ngrid,)
        self._dvolume = dvol_atoms * watoms

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def coord_type(self) -> str:
        return "cart"

    def get_dvolume(self) -> torch.Tensor:
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
    dtype = atompos.dtype
    device = atompos.device

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
