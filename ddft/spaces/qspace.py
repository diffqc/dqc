import torch
import numpy as np
from ddft.spaces.base_space import BaseSpace

class QSpace(BaseSpace):
    """
    Q-space is the Fourier space for real signal in the spatial domain.
    It transforms real-valued signal in spatial domain to complex-valued signal
        in the Fourier domain.
    """
    def __init__(self, rgrid, boxshape):
        # rgrid is (nr,ndim)
        # boxshape is (nx,ny,nz) for 3D, (nx,ny) for 2D, and (nx,) for 1D
        self._rgrid = rgrid
        self._boxshape = boxshape
        self._qgrid, self._qboxshape = _construct_qgrid(rgrid, boxshape)
        self.nr, self.ndim = self._rgrid.shape

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def qgrid(self):
        return self._qgrid

    @property
    def boxshape(self):
        return self._boxshape

    @property
    def qboxshape(self):
        return self._qboxshape

    def transformsig(self, sig, dim=-1):
        # sig: (...,nr,...)

        # normalize the dim
        ndim = sig.ndim
        if dim < 0:
            dim = ndim + dim

        # put the nr at dim=-1
        transposed = (dim != ndim-1)
        if transposed:
            sig = sig.transpose(dim, -1)

        sigbox = self.boxifysig(sig, dim=-1) # (...,...,nx,ny,nz)
        sigftbox = torch.rfft(sigbox, signal_ndim=3) # (...,nx/2,ny/2,nz/2,2)
        sigft = self.flattensig(sigftbox, dim=-2, qdom=True) # (...,nr/8,2)

        if transposed:
            sigft = sigft.transpose(dim,-2)

        return sigft # (...,nr/8,...,2)

    def invtransformsig(self, tsig, dim=-2):
        # tsig: (...,ns,...,2)

        # normalize the dim
        ndim = tsig.ndim
        if dim < 0:
            dim = ndim + dim

        # put the nr at dim=-2
        transposed = (dim != ndim-2)
        if transposed:
            tsig = tsig.transpose(dim, -2) # (...,ns,2)

        tsigbox = self.boxifysig(tsig, dim=-2, qdom=True) #(...,nx/2,ny/2,nz/2,2)
        sigbox = torch.irfft(tsigbox, signal_ndim=3) # (...,nx,ny,nz)
        sig = self.flattensig(sigbox, dim=-1)

        if transposed:
            sig = sig.transpose(dim,-1)
        return sig # (...,nr,...)

def _construct_qgrid_1(xgrid):
    N = len(xgrid)
    dr = xgrid[1] - xgrid[0]
    boxsize = xgrid[-1] - xgrid[0]
    dq = 2*np.pi / boxsize
    Nhalf = (N // 2) + 1
    offset = (N + 1) % 2
    qgrid_half = torch.arange(Nhalf).to(xgrid.dtype).to(xgrid.device)
    return qgrid_half

def _construct_qgrid(rgrid, boxshape):
    # rgrid: (nr, ndim)
    # boxshape = (nx, ny, nz) for 3D

    nr, ndim = rgrid.shape
    rshape = torch.LongTensor(rgrid.shape)
    rgrid = rgrid.view(*boxshape, ndim) # (nx, ny, nz, ndim)
    qgrids = []
    newboxshape = []
    for i in range(ndim):
        index = torch.arange(rgrid.shape[i]) * torch.prod(rshape[i:]) + i
        xgrid = torch.take(rgrid, index=index)
        qgrid = _construct_qgrid_1(xgrid) # (nx,)
        qshape = [qgrid.shape[0] if i==j else 1 for j in range(ndim)] # =(1,ny,1)
        newboxshape.append(qgrid.shape[0])
        qgrids.append(qgrid.view(*qshape))

    newboxshape = tuple(newboxshape)
    for i in range(ndim):
        qgrids[i] = qgrids[i].expand(*newboxshape).unsqueeze(-1) # (nx,ny,nz,1)

    qgrid = torch.cat(qgrids, dim=-1).view(-1,ndim) # (ns,ndim)
    return qgrid, newboxshape
