import torch
import numpy as np
from ddft.hamiltons.base_hamilton import BaseHamilton

class HamiltonPlaneWave(BaseHamilton):
    def __init__(self, space):
        # rgrid is (nr,ndim), ordered by (x, y, z)
        # boxshape (ndim,) = (nx, ny, nz)
        super(HamiltonPlaneWave, self).__init__()

        # set up the space
        self.space = space
        self.qgrid = self.space.qgrid # (ns,ndim)
        self.q2 = (self.qgrid*self.qgrid).sum(dim=-1,keepdim=True).expand(-1,2) # (ns,2)

        rgrid = self.space.rgrid
        boxshape = self.space.boxshape
        self.ndim = self.space.ndim
        nr = rgrid.shape[0]
        self._shape = (nr, nr)

        # get the pixel size
        self.pixsize = rgrid[1,:] - rgrid[0,:] # (ndim,)
        self.dr3 = torch.prod(self.pixsize)
        self.inv_dr3 = 1.0 / self.dr3

        # prepare the diagonal part of kinetics
        self.Kdiag = torch.ones(nr).to(rgrid.dtype).to(rgrid.device) * self.ndim # (nr,)

    def kinetics(self, wf):
        # wf: (nbatch, nr, ncols)
        # wf consists of points in the real space

        # perform the operation in q-space, so FT the wf first
        wfT = wf.transpose(-2, -1) # (nbatch, ncols, nr)
        coeff = self.space.transformsig(wfT, dim=-1) # (nbatch, ncols, ns, 2)

        # multiply with |q|^2 and IFT transform it back
        coeffq2 = -coeff * self.q2 # (nbatch, ncols, ns, 2)
        kin = self.space.invtransformsig(coeffq2, dim=-2) # (nbatch, ncols, nr)

        # revert to the original shape
        return kin.transpose(-2, -1) # (nbatch, nr, ncols)

    def kinetics_diag(self, nbatch):
        return self.Kdiag.unsqueeze(0).expand(nbatch,-1) # (nbatch, nr)

    def getdens(self, eigvec2):
        return eigvec2 * self.inv_dr3

    def integralbox(self, p, dim=-1):
        return p.sum(dim=dim) * self.dr3

    @property
    def shape(self):
        return self._shape
