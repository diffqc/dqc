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

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        eps = 1e-10
        intgn1 = f * self.rs*self.rs * (self.rs + self.rmin)
        int1 = torch.cumsum(intgn1, dim=-1) * self.dlogr
        intgn2 = int1 / (self.rs * self.rs + eps) * (self.rs + self.rmin)
        int2 = torch.cumsum(intgn2, dim=-1) * self.dlogr
        return int2

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape
