import os
import warnings
import torch
import numpy as np
import ddft
from ddft.grids.base_grid import BaseGrid, BaseRadialGrid, BaseRadialAngularGrid
from ddft.utils.spharmonics import spharmonics

class Lebedev(BaseRadialAngularGrid):
    def __init__(self, radgrid, prec, basis_maxangmom=None, dtype=torch.float, device=torch.device('cpu')):
        super(Lebedev, self).__init__()

        # radgrid must be a BaseRadialGrid
        if not isinstance(radgrid, BaseRadialGrid):
            raise TypeError("Argument radgrid must be a BaseRadialGrid")

        # the precision must be an odd number in range [3, 131]
        self.prec = prec
        self.basis_maxangmom = basis_maxangmom if basis_maxangmom is not None else prec
        assert (prec % 2 == 1) and (3 <= prec <= 131),\
               "Precision must be an odd number between 3 and 131"

        self.dtype = dtype
        self.device = device

        # load the datasets from https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
        dset_path = os.path.join(os.path.split(ddft.__file__)[0], "datasets", "lebedevquad", "lebedev_%03d.txt"%prec)
        assert os.path.exists(dset_path), "The dataset lebedev_%03d.txt does not exist" % prec
        lebedev_dsets = torch.tensor(np.loadtxt(dset_path), dtype=dtype, device=device)
        self.phithetargrid = lebedev_dsets[:,:2] / 180.0 * np.pi # (nphitheta,2)
        self.wphitheta = lebedev_dsets[:,-1] # (nphitheta)
        nphitheta = self.phithetargrid.shape[0]

        # get the radial grid
        self.radgrid = radgrid
        self.radrgrid = radgrid.rgrid[:,0] # (nrad,)
        nrad = self.radrgrid.shape[0]
        self.nrad = nrad

        # combine the grids
        # (nrad, nphitheta)
        rg = self.radrgrid.unsqueeze(-1).repeat(1, nphitheta).unsqueeze(-1) # (nrad, nphitheta, 1)
        ptg = self.phithetargrid.unsqueeze(0).repeat(nrad, 1, 1) # (nrad, nphitheta, 2)
        self._rgrid = torch.cat((rg, ptg), dim=-1).view(-1, 3) # (nrad*nphitheta, 3)

        # get the integration part
        self._dvolume_rad = self.radgrid.get_dvolume() # (nrad,)
        # print(self._dvolume_rad.sum()/self.radrgrid.max()**3/(4*np.pi/3), self.wphitheta.sum())
        dvolume = self._dvolume_rad.unsqueeze(-1) * self.wphitheta
        self._dvolume = dvolume.view(-1) # (nrad*nphitheta)

        # # check the basis orthonormality
        # basis = self._get_basis() # (nsh, nphitheta)
        # basis_olp = torch.matmul(basis*self.wphitheta, basis.transpose(-2,-1))
        # assert torch.allclose(basis_olp, torch.eye(basis_olp.shape[0], dtype=basis_olp.dtype, device=basis_olp.device))
        # raise RuntimeError

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # nr = nrad * nphitheta

        # get the spherical harmonics components of f as function of radius
        eps = 1e-12
        nbatch, nr = f.shape
        f1 = f.view(nbatch, self.nrad, -1) # (nbatch, nrad, nphitheta)
        basis = self._get_basis() # (nsh, nphitheta)
        basis_integrate = basis * self.wphitheta
        frad_lm = torch.bmm(basis_integrate.unsqueeze(0).expand(nbatch,-1,-1), f1.transpose(-2,-1)) # (nbatch, nsh, nrad)

        # the computation is done by computing the ratio of rless/rgreat first
        # then integrate it
        # it is done this way to prevent numerical instability, although the
        # computation is slightly more expensive

        # calculate the matrix rless / rgreat
        angmoms = self._get_angmoms().unsqueeze(-1) # (nsh,1)
        angmoms1 = angmoms.unsqueeze(-1)
        rless = torch.min(self.radrgrid.unsqueeze(-1), self.radrgrid) # (nrad, nrad)
        rgreat = torch.max(self.radrgrid.unsqueeze(-1), self.radrgrid)
        rratio = (rless / rgreat)**angmoms1 / rgreat # (nsh, nrad, nrad)

        # the integralbox for radial grid is integral[4*pi*r^2 f(r) dr] while here
        # we only need to do integral[f(r) dr]. That's why it is divided by (4*np.pi)
        # and it is not multiplied with (self.radrgrid**2) in the lines below
        intgn = (frad_lm).unsqueeze(-2) * rratio # (nbatch, nsh, nrad, nrad)
        vrad_lm = self.radgrid.integralbox(intgn / (4*np.pi), dim=-1) / (2*angmoms+1)

        # convert back to the spatial basis
        v = torch.matmul(vrad_lm.transpose(-2,-1), basis) # (nbatch, nrad, nphitheta)
        v = v.view(nbatch, nr)
        return -v

    @property
    def radial_grid(self):
        return self.radgrid

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        warnings.warn("Boxshape is obsolete. Please refrain in using it.")

    def _get_basis(self):
        if not hasattr(self, "_basis_"):
            phi = self.phithetargrid[:,0]
            costheta = torch.cos(self.phithetargrid[:,1])
            self._basis_ = spharmonics(costheta, phi, self.basis_maxangmom)
        return self._basis_

    def _get_angmoms(self):
        if not hasattr(self, "_angmoms_"):
            lhat = []
            for angmom in range(self.basis_maxangmom+1):
                lhat = lhat + [angmom]*(2*angmom+1)
            self._angmoms_ = torch.tensor(lhat, dtype=self.dtype, device=self.device) # (nsh,)
        return self._angmoms_
