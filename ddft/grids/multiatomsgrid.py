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
    * atomradius: torch.tensor (natoms,) or None
        The atom radius. If None, it will be assumed to be all 1.
    * dtype, device:
        Type and device of the tensors involved in the calculations.
    """
    def __init__(self, atomgrid, atompos, atomradius=None, dtype=torch.float, device=torch.device('cpu')):
        super(BeckeMultiGrid, self).__init__()

        # atomgrid must be a 3DGrid
        if not isinstance(atomgrid, Base3DGrid):
            raise TypeError("Argument atomgrid must be a Base3DGrid")

        natoms = atompos.shape[0]
        self.natoms = natoms
        self.atompos = atompos
        self.atomradius = atomradius

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
        return self._atomgrid

    def get_atom_weights(self):
        xyz = self.rgrid_in_xyz # (nr, 3)
        rgatoms = torch.norm(xyz - self.atompos.unsqueeze(1), dim=-1) # (natoms, nr)
        ratoms = torch.norm(self.atompos - self.atompos.unsqueeze(1), dim=-1) # (natoms, natoms)
        mu_ij = (rgatoms - rgatoms.unsqueeze(1)) / (ratoms.unsqueeze(-1) + 1e-12) # (natoms, natoms, nr)

        # calculate the distortion due to heterogeneity
        # (Appendix in Becke's https://doi.org/10.1063/1.454033)
        if self.atomradius is not None:
            chiij = self.atomradius / self.atomradius.unsqueeze(1) # (natoms, natoms)
            uij = (self.atomradius - self.atomradius.unsqueeze(1)) / \
                  (self.atomradius + self.atomradius.unsqueeze(1))
            aij = torch.clamp(uij / (uij*uij - 1), min=-0.45, max=0.45)
            mu_ij = mu_ij + aij * (1-mu_ij*mu_ij)

        f = mu_ij
        for _ in range(3):
            f = 0.5 * f * (3 - f*f)
        # small epsilon to avoid nan in the gradient
        s = 0.5 * (1.+1e-12 - f) # (natoms, natoms, nr)
        s = s + 0.5*torch.eye(self.natoms).unsqueeze(-1)
        p = s.prod(dim=0) # (natoms, nr)
        p = p / p.sum(dim=0, keepdim=True) # (natoms, nr)

        watoms0 = p.view(self.natoms, self.natoms, -1) # (natoms, natoms, ngrid)
        watoms = watoms0.diagonal(dim1=0, dim2=1).transpose(-2,-1).contiguous() # (natoms, ngrid)
        return watoms

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # split the f first
        nbatch = f.shape[0]
        fatoms = f.view(nbatch, self.natoms, -1) * self.get_atom_weights() # (nbatch, natoms, ngrid)
        natoms = self.atom_grid.integralbox(-fatoms / (4*np.pi), dim=-1) # (nbatch, natoms)
        fatoms = fatoms.contiguous().view(-1, fatoms.shape[-1]) # (nbatch*natoms, ngrid)

        Vatoms = self.atom_grid.solve_poisson(fatoms).view(nbatch, self.natoms, -1) # (nbatch, natoms, ngrid)
        def get_extrap_fcn(iatom):
            natom = natoms[:,iatom] # (nbatch,)
            # rgrid: (nrextrap, ndim)
            extrapfcn = lambda rgrid: natom.unsqueeze(-1) / (rgrid[:,0] + 1e-12)
            return extrapfcn

        # get the grid outside the original grid for the indexed atom
        def get_outside_rgrid(iatom):
            rgrid = self._rgrid.view(self.natoms, -1, self._rgrid.shape[-1]) # (natoms, ngrid, ndim)
            res = torch.cat((rgrid[:iatom,:,:], rgrid[iatom+1:,:,:]), dim=0) # (natoms-1, ngrid, ndim)
            return res.view(-1, res.shape[-1]) # ((natoms-1) * ngrid, ndim)

        if self.natoms == 1:
            return Vatoms.view(Vatoms.shape[0], -1) # (nbatch, natoms*ngrid)

        # combine the potentials with interpolation and extrapolation
        Vtot = torch.zeros_like(Vatoms).to(Vatoms.device).view(nbatch, -1) # (nbatch, natoms*ngrid)
        for i in range(self.natoms):
            gridxyz = get_outside_rgrid(i) - self.atompos[i,:] # ((natoms-1)*ngrid, 3)
            gridi = self.atom_grid.xyz_to_rgrid(gridxyz)
            Vinterp = self.atom_grid.interpolate(Vatoms[:,i,:], gridi,
                extrap=get_extrap_fcn(i)) # (nbatch, (natoms-1)*ngrid)
            Vinterp = Vinterp.view(Vinterp.shape[0], self.natoms-1, -1) # (nbatch, natoms-1, ngrid)

            # combine the interpolated function with the original function
            Vinterp = torch.cat((Vinterp[:,:i,:], Vatoms[:,i:i+1,:], Vinterp[:,i:,:]), dim=1).view(Vinterp.shape[0], -1)
            Vtot += Vinterp

        return Vtot

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def rgrid_in_xyz(self):
        return self._rgrid

    @property
    def boxshape(self):
        warnings.warn("Boxshape is obsolete. Please refrain in using it.")
