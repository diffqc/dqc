import torch
import numpy as np
from ddft.hamiltons.base_hamilton import BaseHamilton

class HamiltonPlaneWave(BaseHamilton):
    def __init__(self, space):
        # rgrid is (nr,ndim), ordered by (x, y, z)
        # boxshape (ndim,) = (nx, ny, nz)
        nr = len(space.rgrid)
        super(HamiltonPlaneWave, self).__init__(
            shape = (nr, nr),
            is_symmetric = True,
            is_real = True)

        # set up the space
        self.space = space
        self.qgrid = self.space.qgrid # (ns,ndim)
        self.q2 = (self.qgrid*self.qgrid).sum(dim=-1, keepdim=True) # (ns,1)

        rgrid = self.space.rgrid # (nr, ndim)
        boxshape = self.space.boxshape # (nx,ny,nz)
        self.ndim = self.space.ndim
        nr, self.ndim = rgrid.shape

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

    def forward(self, wf, vext):
        # wf: (nbatch, nr, ncols)
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

        yq = self.space.transformsig(y, dim=1) # (nbatch, ns, ncols)
        nbatch, ns, ncols = yq.shape

        # get the diagonal and apply the inverse
        diag_kin = self.q2.squeeze(-1).unsqueeze(0).expand(nbatch, -1) * 0.5 # (nbatch,ns)
        sumvext = vext.sum(dim=-1, keepdim=True) / (nr*1.0)
        sumvext = sumvext.expand(-1, ns) # (nbatch, ns)
        diag = diag_kin + sumvext # (nbatch, ns)
        diag = diag.unsqueeze(-1) # (nbatch, ns, 1)
        if biases is not None:
            diag = diag - biases.unsqueeze(1) # (nbatch, nr, ncols)

        # invert the diagonal
        diag[diag.abs() < 1e-6] = 1e-6
        yq2 = yq / diag

        yres = self.space.invtransformsig(yq2, dim=1)
        return yres

    def getdens(self, eigvecs):
        # eigvecs: (nbatch, nr, nlowest)
        dens = (eigvecs * eigvecs)
        sumdens = self.integralbox(dens, dim=1).unsqueeze(1)
        return dens / sumdens

    def integralbox(self, p, dim=-1):
        return p.sum(dim=dim) * self.dr3

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ddft.spaces.qspace import QSpace

    dtype = torch.float64
    ndim = 3
    boxshape = [51, 51, 51][:ndim]
    boxsizes = [10.0, 10.0, 10.0][:ndim]
    rgrids = [torch.linspace(-boxsize/2., boxsize/2., nx+1)[:-1].to(dtype) for (boxsize,nx) in zip(boxsizes,boxshape)]
    rgrids = torch.meshgrid(*rgrids) # (nx,ny,nz)
    rgrid = torch.cat([rgrid.unsqueeze(-1) for rgrid in rgrids], dim=-1).view(-1,ndim) # (nr,3)
    nr = rgrid.shape[0]

    qspace = QSpace(rgrid, boxshape)
    hpw = HamiltonPlaneWave(qspace)

    nbatch = 1
    vext = torch.ones((nbatch, nr), dtype=dtype) * 1
    wf = torch.rand((nbatch, nr, 1), dtype=dtype)
    hr = hpw(wf, vext)
    wf_retr = hpw.precond(hr, vext)

    dev_wf = (wf - wf_retr).squeeze()
    print(dev_wf)
    plt.plot(dev_wf.numpy())
    plt.show()
