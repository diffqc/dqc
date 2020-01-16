import torch
import numpy as np
from ddft.hamiltons.base_hamilton import BaseHamilton
from ddft.hamiltons.hspatial1d import HamiltonSpatial1D

class HamiltonPW1D(HamiltonSpatial1D):
    def __init__(self, rgrid):
        super(HamiltonPW1D, self).__init__(rgrid)

        # construct the r-grid and q-grid
        N = len(rgrid)
        dr = rgrid[1] - rgrid[0]
        boxsize = rgrid[-1] - rgrid[0]
        dq = 2*np.pi / boxsize
        Nhalf = (N // 2) + 1
        offset = (N + 1) % 2
        qgrid_half = torch.arange(Nhalf)
        self.qgrid = qgrid_half
        self.q2 = self.qgrid*self.qgrid

    def kinetics(self, wf):
        # wf: (nbatch, nr, ncols)
        # wf consists of points in the real space

        # perform the operation in q-space, so FT the wf first
        wfT = wf.transpose(-2, -1) # (nbatch, ncols, nr)
        coeff = torch.rfft(wfT, signal_ndim=1) # (nbatch, ncols, nr, 2)

        # multiply with |q|^2 and IFT transform it back
        q2 = self.q2.unsqueeze(-1).expand(-1,2) # (nr, 2)
        coeffq2 = coeff * q2
        kin = torch.irfft(coeffq2, signal_ndim=1) # (nbatch, ncols, nr)

        # revert to the original shape
        return kin.transpose(-2, -1) # (nbatch, nr, ncols)
