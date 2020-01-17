import torch
import numpy as np
from ddft.hamiltons.base_hamilton import BaseHamilton

class HamiltonPW3D(BaseHamilton):
    def __init__(self, rgrid, boxshape):
        # rgrid is (nr,3), ordered by (x, y, z)
        # boxshape (3,) = (nx, ny, nz)
        super(HamiltonPW3D, self).__init__()
        self._rgrid = rgrid
        self._boxshape = boxshape
        self.ndim = 3
        nr = rgrid.shape[0]

        # get the pixel size
        self.pixsize = rgrid[1,:] - rgrid[0,:] # (3,)
        self.dr3 = torch.prod(self.pixsize)
        self.inv_dr3 = 1.0 / self.dr3

        # check the shape
        if torch.prod(boxshape) != nr:
            msg = "The product of boxshape elements must be equal to the "\
                  "first dimension of rgrid"
            raise ValueError(msg)

        # construct the q-grid
        self.qhalf = _construct_qgrid(qgrid, boxshape) # (nr,3)

        # qhalf2_mag is the |q|^2 multiplier for real and complex parts
        self.qhalf2_mag = (self.qhalf * self.qhalf).sum(dim=-1)\
                .view(*self.boxshape).unsqueeze(-1).expand(-1,-1,-1,2) # (nx, ny, nz, 2)

        # prepare the diagonal part of kinetics
        self.Kdiag = torch.ones(nr).to(rgrid.dtype).to(rgrid.device) * self.ndim # (nr,)

    def kinetics(self, wf):
        # wf: (nbatch, nr, ncols)
        # wf consists of points in the real space

        # perform the operation in q-space, so FT the wf first
        wfT = wf.transpose(-2, -1) # (nbatch, ncols, nr)
        wfT = self.boxifysig(wfT, dim=-1) # (nbatch, ncols, nx, ny, nz)
        coeff = torch.rfft(wfT, signal_ndim=3) # (nbatch, ncols, nx, ny, nz, 2)

        # multiply with |q|^2 and IFT transform it back
        coeffq2 = coeff * self.qhalf2_mag # (nbatch, ncols, nx, ny, nz, 2)
        kin = torch.irfft(coeffq2, signal_ndim=3) # (nbatch, ncols, nx, ny, nz)
        kin = self.flattensig(kin, dim=-1) # (nbatch, ncols, nr)

        # revert to the original shape
        return kin.transpose(-2, -1) # (nbatch, nr, ncols)

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape

    def kinetics_diag(self, nbatch):
        return self.Kdiag.unsqueeze(0).expand(nbatch,-1) # (nbatch, nr)

    def getdens(self, eigvec2):
        return eigvec2 * self.inv_dr3

    def integralbox(self, p, dim=-1):
        return p.sum(dim=dim) * self.dr3

def _construct_qgrid(rgrid, boxshape):
    # rgrid: (nr, 3)
    # boxshape = (nx, ny, nz)

    nx, ny, nz = boxshape
    rgrid = rgrid.view(*boxshape) # (nx, ny, nz, 3)
    xgrid = rgrid[:,0,0,0]
    ygrid = rgrid[0,:,0,1]
    zgrid = rgrid[0,0,:,2]
    qxgrid = _construct_qgrid_1(xgrid) # (nx,)
    qygrid = _construct_qgrid_1(ygrid) # (ny,)
    qzgrid = _construct_qgrid_1(zgrid) # (nz,)

    # reshape q-grid
    qxgrid2 = qxgrid.unsqueeze(-1).unsqueeze(-1).expand(nx,ny,nz) # (nx,ny,nz)
    qygrid2 = qxgrid.unsqueeze(-1).unsqueeze( 0).expand(nx,ny,nz) # (nx,ny,nz)
    qzgrid2 = qxgrid.unsqueeze( 0).unsqueeze( 0).expand(nx,ny,nz) # (nx,ny,nz)
    qgrid = torch.cat([
        qxgrid2.unsqueeze(-1),
        qygrid2.unsqueeze(-1),
        qzgrid2.unsqueeze(-1),
    ], dim=-1) # (nx, ny, nz, 3)
    return qgrid

def _construct_qgrid_1(xgrid):
    N = len(xgrid)
    dr = xgrid[1] - xgrid[0]
    boxsize = xgrid[-1] - xgrid[0]
    dq = 2*np.pi / boxsize
    Nhalf = (N // 2) + 1
    offset = (N + 1) % 2
    qgrid_half = torch.arange(Nhalf).to(xgrid.dtype).to(xgrid.device)
    return qgrid_half
