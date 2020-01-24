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
        self.q2 = (self.qgrid*self.qgrid).sum(dim=-1, keepdim=True) # (ns,1)

        rgrid = self.space.rgrid # (nr, ndim)
        boxshape = self.space.boxshape # (nx,ny,nz)
        self.ndim = self.space.ndim
        ns, ndim = self.qgrid.shape
        self._shape = (ns, ns)

        # get the pixel size
        idx = 0
        allshape = (*boxshape, rgrid.shape[-1])
        m = 1
        for i in range(ndim,0,-1):
            m *= allshape[i]
            idx += m
        self.pixsize = rgrid[idx,:] - rgrid[0,:] # (ndim,)
        self.dr3 = torch.prod(self.pixsize)
        self.inv_dr3 = 1.0 / self.dr3

    def apply(self, wf, vext, *params):
        # wf: (nbatch, ns, ncols)
        # vext: (nbatch, nr)

        # the kinetics part is just multiply wf with q^2
        kin = 0.5 * wf * self.q2 # (nbatch, ns, ncols)

        # the potential part is element-wise multiplication in spatial domain
        # so we need to transform wf to spatial domain first
        wfr = self.space.invtransformsig(wf, dim=1) # (nbatch, nr, ncols)
        potr = wfr * vext.unsqueeze(-1) # (nbatch, nr, ncols)
        pot = self.space.transformsig(potr, dim=1) # (nbatch, ns, ncols)

        h = kin+pot # (nbatch, ns, ncols)
        return h

    def applyT(self, wf, vext, *params):
        # wf: (nbatch, nr, ncols)
        # vext: (nbatch, nr)

        # the kinetics part is the same as the forward part
        kin = 0.5 * wf * self.q2

        # the potential part is just the hermitian of the forward part
        wfr = self.space.Ttransformsig(wf, dim=1)
        potr = wfr * vext.unsqueeze(-1)
        pot = self.space.invTtransform(potr, dim=1)

        hH = kin + pot
        return hH

    def diag(self, vext):
        # vext: (nbatch, nr)
        nbatch, nr = vext.shape

        # the diagonal from kinetics part: self.q2
        kin = self.q2.squeeze(-1).unsqueeze(0).expand(nbatch,-1) # (nbatch, ns)
        ns = kin.shape[1]

        # diagonal from potential part: sum(vext) / sqrt(N)
        sumvext = vext.sum(dim=-1, keepdim=True) / np.sqrt(nr * 1.0) # (nbatch,1)
        sumvext = sumvext.expand(-1,ns)

        return kin + sumvext

    def getdens(self, eigvecs):
        # eigvecs: (nbatch, ns, nlowest)
        evr = self.space.invtransformsig(eigvecs, dim=1) # (nbatch, nr, nlowest)
        densflat = (evr*evr) # (nbatch, nr, nlowest)
        sumdens = self.integralbox(densflat, dim=1).unsqueeze(1) # (nbatch, 1, nlowest)
        return densflat / sumdens

    def integralbox(self, p, dim=-1):
        return p.sum(dim=dim) * self.dr3

    @property
    def shape(self):
        return self._shape

    @property
    def issymmetric(self):
        return self.space.isorthogonal
