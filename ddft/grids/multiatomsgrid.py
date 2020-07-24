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
    * atomgrid: Base3DGrid or list of Base3DGrid
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
        if not isinstance(atomgrid, Base3DGrid) and not hasattr(atomgrid, "__iter__"):
            raise TypeError("Argument atomgrid must be a Base3DGrid or a list of Base3DGrid")

        natoms = atompos.shape[0]
        self.natoms = natoms
        self.atompos = atompos
        self.atomradius = atomradius

        if isinstance(atomgrid, Base3DGrid):
            self.same_grid = True
            self._atomgrids = [atomgrid for _ in range(natoms)]
        else:
            self.same_grid = False
            self._atomgrids = atomgrid

        # construct the size of each grids
        self.ngrids = np.asarray([gr.rgrid.shape[0] for gr in self._atomgrids])
        self.idx_grids_r = np.cumsum(self.ngrids)
        self.idx_grids_l = self.idx_grids_r - self.ngrids
        self.same_sizes = np.all(self.ngrids == self.ngrids[0])

        # obtain the grid position
        # list of natoms (ngrid, 3)
        self._rgrid = [(gr.rgrid_in_xyz + pos) for (gr,pos) in zip(self._atomgrids, atompos)]
        self._rgrid = torch.cat(self._rgrid, dim=0) # (nr, 3)

        # obtain the dvolume
        dvolume_atoms = torch.cat([gr.get_dvolume() for gr in self._atomgrids], dim=0) # (nr,)
        weights_atoms = self.get_atom_weights() # (nr,)
        self._dvolume = dvolume_atoms * weights_atoms

    @property
    def atom_grids(self):
        return self._atomgrids

    def get_atom_weights(self):
        xyz = self.rgrid_in_xyz # (nr, 3)
        rgatoms = torch.norm(xyz - self.atompos.unsqueeze(1), dim=-1) # (natoms, nr)
        rdatoms = self.atompos - self.atompos.unsqueeze(1) # (natoms, natoms, ndim)
        # add the diagonal to stabilize the gradient calculation
        rdatoms = rdatoms + torch.eye(rdatoms.shape[0], dtype=rdatoms.dtype, device=rdatoms.device).unsqueeze(-1)
        ratoms = torch.norm(rdatoms, dim=-1) # (natoms, natoms)
        mu_ij = (rgatoms - rgatoms.unsqueeze(1)) / ratoms.unsqueeze(-1) # (natoms, natoms, nr)

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

        if self.same_sizes:
            watoms0 = p.view(self.natoms, self.natoms, -1) # (natoms, natoms, ngrid)
            watoms = watoms0.diagonal(dim1=0, dim2=1).transpose(-2,-1).contiguous().view(-1) # (natoms*ngrid)
        else:
            watoms_list = []
            for i in range(self.natoms):
                watoms_list.append(p[i,self.idx_grids_l[i]:self.idx_grids_r[i]]) # (ngrid)
            watoms = torch.cat(watoms_list, dim=0) # (nr,)
        return watoms

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # split the f first
        nbatch = f.shape[0]
        fatoms = f * self.get_atom_weights() # (nbatch, nr)
        fatoms_list = []
        for i in range(self.natoms):
            fatoms_list.append(fatoms[:,self.idx_grids_l[i]:self.idx_grids_r[i]]) # (nbatch, ngrid)

        natoms_list = []
        Vatoms_list = []
        for (agr,fa) in zip(self.atom_grids, fatoms_list):
            natoms_list.append(agr.integralbox(-fa/(4*np.pi), dim=-1)) # (nbatch,)
            Vatoms_list.append(agr.solve_poisson(fa)) # (nbatch, ngrid)

        def get_extrap_fcn(iatom):
            natom = natoms_list[iatom] # (nbatch,)
            # rgrid: (nrextrap, ndim)
            extrapfcn = lambda rgrid: natom.unsqueeze(-1) / (rgrid[:,0] + 1e-12)
            return extrapfcn

        # get the grid outside the original grid for the indexed atom
        def get_outside_rgrid(iatom):
            rgrid = self._rgrid # (nr, ndim)
            res = torch.cat((rgrid[:self.idx_grids_l[iatom],:], rgrid[self.idx_grids_r[iatom]:,:]), dim=0)
            return res

        if self.natoms == 1:
            return Vatoms_list[0]

        # perform interpolation and extrapolation
        Vtot = torch.zeros_like(fatoms).to(fatoms.device) # (nbatch, nr)
        for i in range(self.natoms):
            agr = self.atom_grids[i]
            gridxyz = get_outside_rgrid(i) - self.atompos[i,:] # ((natoms-1)*ngrid, ndim)
            gridi = agr.xyz_to_rgrid(gridxyz)
            Vinterp = agr.interpolate(Vatoms_list[i], gridi,
                extrap=get_extrap_fcn(i)) # (nbatch, (natoms-1)*ngrid)

            idxl = self.idx_grids_l[i]
            Vinterp = torch.cat(
                (Vinterp[:,:idxl], Vatoms_list[i], Vinterp[:,idxl:]), dim=1)
            Vtot = Vtot + Vinterp
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

    #################### editable module parts ####################
    def getparams(self, methodname):
        if methodname == "solve_poisson":
            # return [self.atompos, self._atomgrid.phithetargrid,
            #         self._atomgrid.wphitheta, self._atomgrid.radgrid._dvolume,
            #         self._atomgrid.radrgrid, self._rgrid]
            if self.same_grid:
                return [self.atompos, self._rgrid] + \
                        self.atom_grids[0].getparams("get_dvolume") + \
                        self.atom_grids[0].getparams("solve_poisson") + \
                        self.atom_grids[0].getparams("interpolate")
            else:
                raise RuntimeError("Unimplemented")
        elif methodname == "get_dvolume":
            return [self._dvolume]
        else:
            return super().getparams(methodname)

    def setparams(self, methodname, *params):
        if methodname == "solve_poisson":
            idx = 2
            self.atompos, self._rgrid = params[:idx]
            if self.same_grid:
                idx += self.atom_grids[0].setparams("get_dvolume", *params[idx:])
                idx += self.atom_grids[0].setparams("solve_poisson", *params[idx:])
                idx += self.atom_grids[0].setparams("interpolate", *params[idx:])
            else:
                raise RuntimeError("Unimplemented")
            return idx
        elif methodname == "get_dvolume":
            self._dvolume, = params[:1]
            return 1
        else:
            return super().setparams(methodname, *params)

if __name__ == "__main__":
    import lintorch as lt
    from ddft.grids.radialgrid import LegendreShiftExpRadGrid
    from ddft.grids.sphangulargrid import Lebedev
    dtype = torch.float64
    atompos = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    radgrid = LegendreShiftExpRadGrid(100, 1e-4, 1e2, dtype=dtype)
    anggrid = Lebedev(radgrid, prec=5, basis_maxangmom=4, dtype=dtype)
    grid = BeckeMultiGrid(anggrid, atompos, dtype=dtype)
    rgrid = grid.rgrid.clone().detach()
    f = torch.exp(-rgrid[:,0].unsqueeze(0)**2*0.5)

    lt.list_operating_params(grid.solve_poisson, f)
    lt.list_operating_params(grid.get_dvolume)
