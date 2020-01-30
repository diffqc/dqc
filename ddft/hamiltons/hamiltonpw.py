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
        nr, self.ndim = rgrid.shape
        self._shape = (nr, nr)

        # get the pixel size
        idx = 0
        allshape = (*boxshape, rgrid.shape[-1])
        m = 1
        for i in range(self.ndim,0,-1):
            m *= allshape[i]
            idx += m
        self.pixsize = rgrid[idx,:] - rgrid[0,:] # (ndim,)
        self.dr3 = torch.prod(self.pixsize)
        self.inv_dr3 = 1.0 / self.dr3

    def apply(self, wf, vext, *params):
        # wf: (nbatch, ns, ncols)
        # vext: (nbatch, nr)

        # the kinetics part is q2 in qspace
        wfq = self.space.transformsig(wf, dim=1)
        kinq = 0.5 * wfq * self.q2
        kin = self.space.invtransformsig(kinq, dim=1)

        # the potential is just pointwise multiplication
        pot = wf * vext.unsqueeze(-1)

        return kin+pot

    def precond(self, y, vext, biases=None):
        # y: (nbatch, nr, ncols)
        # vext: (nbatch, nr)
        # biases: (nbatch, ncols) or None

        nbatch, nr, ncols = y.shape

        yq = self.space.transformsig(y, dim=1)

        # get the diagonal and apply the inverse
        diag_kin = self.q2.squeeze(-1).unsqueeze(0).expand(nbatch, -1) # (nbatch,nr)
        sumvext = vext.sum(dim=-1, keepdim=True) / np.sqrt(nr*1.0)
        sumvext = sumvext.expand(-1, nr) # (nbatch, nr)
        diag = diag_kin + sumvext # (nbatch, nr)
        diag = diag.unsqueeze(-1) # (nbatch, nr, 1)
        if biases is not None:
            diag = diag - biases.unsqueeze(1) # (nbatch, nr, ncols)

        # invert the diagonal
        diag[diag.abs() < 1e-6] = 1e-6
        yq2 = yq / diag

        yres = self.space.invtransformsig(yq2, dim=1)
        return yres

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
        # eigvecs: (nbatch, nr, nlowest)
        dens = (eigvecs * eigvecs)
        sumdens = self.integralbox(dens, dim=1).unsqueeze(1)
        return dens / sumdens

    def integralbox(self, p, dim=-1):
        return p.sum(dim=dim) * self.dr3

    @property
    def shape(self):
        return self._shape
