import torch
import numpy as np
from ddft.hamiltons.base_hamilton import BaseHamilton
from ddft.hamiltons.hspatial1d import HamiltonSpatial1D
from ddft.spaces.qspace import QSpace

class HamiltonPW1D(HamiltonSpatial1D):
    def __init__(self, rgrid):
        super(HamiltonPW1D, self).__init__(rgrid)

        self.space = QSpace(rgrid.unsqueeze(-1), (rgrid.shape[0],))
        self.qgrid = self.space.qgrid.squeeze(-1) # (ns,)
        self.q2 = (self.qgrid*self.qgrid).unsqueeze(-1).expand(-1,2) # (ns,2)

    def kinetics(self, wf):
        # wf: (nbatch, nr, ncols)
        # wf consists of points in the real space

        # perform the operation in q-space, so FT the wf first
        wfT = wf.transpose(-2, -1) # (nbatch, ncols, nr)
        coeff = self.space.transformsig(wfT, dim=-1) # (nbatch, ncols, ns, 2)

        # multiply with |q|^2 and IFT transform it back
        coeffq2 = coeff * self.q2 # (nbatch, ncols, ns, 2)
        kin = self.space.invtransformsig(coeffq2, dim=-2) # (nbatch, ncols, nr)

        # revert to the original shape
        return kin.transpose(-2, -1) # (nbatch, nr, ncols)
