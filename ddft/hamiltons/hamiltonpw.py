import torch
import numpy as np
from ddft.hamiltons.base_hamilton import BaseHamilton
from ddft.spaces.qspace import QSpace

class HamiltonPlaneWave(BaseHamilton):
    # Note: even though the name is HamiltonPlaneWave, the basis is spatial
    # basis. We use plane wave just to calculate the kinetics part.

    def __init__(self, grid):
        self.space = QSpace(grid.rgrid, grid.boxshape)
        self._grid = grid
        nr = len(grid.rgrid)
        super(HamiltonPlaneWave, self).__init__(
            shape = (nr, nr),
            is_symmetric = True,
            is_real = True)

        # set up the qspace
        self.qgrid = self.space.qgrid # (ns,ndim)
        self.q2 = (self.qgrid*self.qgrid).sum(dim=-1, keepdim=True) # (ns,1)

        rgrid = grid.rgrid # (nr, ndim)
        boxshape = grid.boxshape # (nx,ny,nz)
        self.ndim = self.space.ndim
        nr, self.ndim = rgrid.shape

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

    def precond(self, y, vext, biases=None, M=None, mparams=None):
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

    def torgrid(self, wfs, dim=-2):
        return wfs

    @property
    def grid(self):
        return self._grid

    def getvhartree(self, dens):
        raise RuntimeError("getvhartree for HamiltonPlaneWave has not been implemented")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ddft.grids.linearnd import LinearNDGrid

    dtype = torch.float64
    ndim = 3
    boxshape = torch.tensor([51, 51, 51][:ndim])
    boxsizes = torch.tensor([10.0, 10.0, 10.0][:ndim], dtype=dtype)

    grid = LinearNDGrid(boxsizes, boxshape)
    hpw = HamiltonPlaneWave(grid)

    nbatch = 1
    nr = grid.rgrid.shape[0]
    vext = torch.ones((nbatch, nr), dtype=dtype) * 1
    wf = torch.rand((nbatch, nr, 1), dtype=dtype)
    hr = hpw(wf, vext)
    wf_retr = hpw.precond(hr, vext)

    dev_wf = (wf - wf_retr).squeeze()
    print(dev_wf)
    plt.plot(dev_wf.numpy())
    plt.show()
