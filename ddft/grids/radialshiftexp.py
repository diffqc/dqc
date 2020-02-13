import torch
import numpy as np
from ddft.grids.base_grid import BaseGrid

class RadialShiftExp(BaseGrid):
    def __init__(self, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
        logr = torch.linspace(np.log(rmin), np.log(rmax), nr).to(dtype).to(device)
        unshifted_rgrid = torch.exp(logr)
        self._boxshape = (nr,)
        self.rmin = rmin
        self.rs = unshifted_rgrid - self.rmin
        self._rgrid = self.rs.unsqueeze(1) # (nr, 1)
        self.dlogr = logr[1] - logr[0]

    def get_integrand_box(self, p):
        return p * (self.rs + self.rmin) * 4 * np.pi * self.rs*self.rs * self.dlogr

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape
