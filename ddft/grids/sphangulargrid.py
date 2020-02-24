import os
import warnings
import torch
import numpy as np
import ddft
from ddft.grids.base_grid import BaseGrid

class Lebedev(BaseGrid):
    def __init__(self, radgrid, prec, dtype=torch.float, device=torch.device('cpu')):
        super(Lebedev, self).__init__()

        # the precision must be an odd number in range [3, 131]
        assert (prec % 2 == 1) and (3 <= prec <= 131),\
               "Precision must be an odd number between 3 and 131"

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

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        pass
        # ???

    @property
    def radial_grid(self):
        return self.radgrid

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        warnings.warn("Boxshape is obsolete. Please refrain in using it.")
