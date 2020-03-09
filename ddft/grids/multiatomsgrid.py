import os
import warnings
import torch
import numpy as np
import ddft
from ddft.grids.base_grid import BaseRadialAngularGrid, BaseMultiAtomsGrid, Base3DGrid
from ddft.utils.spharmonics import spharmonics

class BeckeMultiGrid(BaseMultiAtomsGrid):
    """
    Using the Becke weighting to split a profile.

    Arguments
    ---------
    * atomgrid: Base3DGrid
        The grid for each individual atom.
    * atompos: torch.tensor (natoms, 3)
        The position of each atom.
    * dtype, device:
        Type and device of the tensors involved in the calculations.
    """
    def __init__(self, atomgrid, atompos, dtype=torch.float, device=torch.device('cpu')):
        super(SSWeights, self).__init__()

        # atomgrid must be a 3DGrid
        if not isinstance(atomgrid, Base3DGrid):
            raise TypeError("Argument atomgrid must be a Base3DGrid")

        natoms = atompos.shape[0]
        self.natoms = natoms
        self.atompos = atompos

        # obtain the grid position
        self._atomgrid = atomgrid
        rgrid_atom = atomgrid.rgrid_in_xyz # (ngrid, 3)
        rgrid = rgrid_atom + atompos.unsqueeze(1) # (natoms, ngrid, 3)
        self._rgrid = rgrid.view(-1, rgrid.shape[-1]) # (natoms*ngrid, 3)

        # obtain the dvolume
        dvolume_atom = atomgrid.get_dvolume().repeat(natoms) # (natoms*ngrid,)
        weights_atom = self.get_atom_weights().view(-1) # (natoms*ngrid,)
        self._dvolume = dvolume_atom * weights_atom

    @property
    def atom_grid(self):
        return _atomgrid

    def get_atom_weights(self):
        xyz = self.rgrid_in_xyz # (nr, 3)
        rgatoms = torch.norm(xyz - self.atompos.unsqueeze(1), dim=-1) # (natoms, nr)
        ratoms = torch.norm(self.atompos - self.atompos.unsqueeze(1), dim=-1) # (natoms, natoms)
        mu_ij = (rgatoms - rgatoms.unsqueeze(1)) / ratoms.unsqueeze(-1) # (natoms, natoms, nr)

        f = mu_ij
        for _ in range(3):
            f = 0.5 * f * (3 - f*f)
        s = 0.5 * (1.0 - f) # (natoms, natoms, nr)
        p = s.prod(dim=0) # (natoms, nr)
        p = p / p.sum(dim=0, keepdim=True) # (natoms, nr)

        watoms = p.view(self.natoms, self.natoms, -1) # (natoms, natoms, ngrid)
        return watoms.diagonal(dim0=0, dim1=1) # (natoms, ngrid)

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # split the f first
        fatoms = f.view(nbatch, self.natoms, -1) * self.get_atom_weights() # (nbatch, natoms, ngrid)
        fatoms = fatoms.view(-1, fatoms.shape[-1]) # (nbatch*natoms, ngrid)

        Vatoms = self.atom_grid.solve_poisson(fatoms) # (nbatch*natoms, ngrid)
        # how to merge it back to V?
        pass

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def rgrid_in_xyz(self):
        return self._rgrid

    @property
    def boxshape(self):
        warnings.warn("Boxshape is obsolete. Please refrain in using it.")
