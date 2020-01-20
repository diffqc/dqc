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
        # qgrid is (ns=nr, ndim)
        # qboxshape is (nx,ny,nz,2)
        self._rgrid = rgrid
        self._boxshape = boxshape
        self._qgrid, self._qboxshape = _construct_qgrid(rgrid, boxshape)
        self.nr = self._rgrid.shape[0]

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

    def transformsig(self, sig, dim=-1, rcomplex=False):
        # sig: (...,nr,...) or (...,nr,2,...)

        # normalize the dim
        ndim = sig.ndim
        if dim < 0:
            dim = ndim + dim

        if rcomplex:
            sig = sig.view(*sig.shape[:dim], self.nr*2, *sig.shape[dim+2:]) # (...,nr*2,...)

        # put the nr at dim=-1
        transposed = (dim != ndim-1)
        if transposed:
            sig = sig.transpose(dim, -1)

        if rcomplex:
            sig = sig.view(*sig.shape[:-1], self.nr, 2) # (...,nr,2)
            sigbox = self.boxifysig(sig, dim=-2) # (...,nx,ny,nz,2)
            sigftbox = torch.fft(sigbox, signal_ndim=self.ndim, normalized=True) # (...,nx,ny,nz,2)
        else:
            sigbox = self.boxifysig(sig, dim=-1) # (...,...,nx,ny,nz)
            sigftbox = torch.rfft(sigbox, signal_ndim=self.ndim, onesided=False, normalized=True) # (...,nx,ny,nz,2)
        sigft = self.flattensig(sigftbox, dim=-1, qdom=True) # (...,ns)

        if transposed:
            sigft = sigft.transpose(dim,-1)

        return sigft # (...,ns,...)

    def invtransformsig(self, tsig, dim=-1, rcomplex=False):
        # tsig: (...,ns,...)
        # return: (...,nr,...) or (...,nr,2,...) if rcomplex

        # normalize the dim
        ndim = tsig.ndim
        if dim < 0:
            dim = ndim + dim

        # put the nr at dim=-1
        transposed = (dim != ndim-1)
        if transposed:
            tsig = tsig.transpose(dim, -1) # (...,ns)

        tsigbox = self.boxifysig(tsig, dim=-1, qdom=True) #(...,nx,ny,nz,2)
        if rcomplex:
            sigbox = torch.ifft(tsigbox, signal_ndim=self.ndim, normalized=True) # (...,nx,ny,nz,2)
            sig = self.flattensig(sigbox, dim=-2) # (...,nr,2)
            sig = sig.view(*sig.shape[:-2], self.nr*2) # (...,nr*2)
        else:
            sigbox = torch.irfft(tsigbox, signal_ndim=self.ndim, onesided=False, normalized=True) # (...,nx,ny,nz)
            sig = self.flattensig(sigbox, dim=-1) # (...,nr)

        if transposed:
            sig = sig.transpose(dim,-1)

        if rcomplex:
            sig = sig.view(*sig.shape[:dim], self.nr, 2, *sig.shape[dim+1:])
        return sig # (...,nr,...) or (...,nr,2,...)

def _construct_qgrid_1(xgrid, full=False):
    N = len(xgrid)
    dr = xgrid[1] - xgrid[0]
    boxsize = xgrid[-1] - xgrid[0]
    dq = 2*np.pi / (N * dr)
    Nhalf = (N // 2) + 1
    offset = (N + 1) % 2
    qgrid_lhalf = torch.arange(Nhalf)
    qgrid_rhalf = -torch.arange(Nhalf-offset-1,0,-1)
    if full:
        qgrid = torch.cat((qgrid_lhalf, qgrid_rhalf))
    else:
        qgrid = qgrid_lhalf
    qgrid = qgrid.to(xgrid.dtype).to(xgrid.device) * dq
    return qgrid

def _construct_qgrid(rgrid, boxshape):
    # rgrid: (nr, ndim)
    # boxshape = (nx, ny, nz) for 3D

    nr, ndim = rgrid.shape
    rgrid = rgrid.view(*boxshape, ndim) # (nx, ny, nz, ndim)
    qgrids = []
    newboxshape = []
    spacing = nr * ndim
    for i in range(ndim):
        spacing = spacing // boxshape[i]
        index = torch.arange(rgrid.shape[i]) * spacing + i
        xgrid = torch.take(rgrid, index=index)
        qgrid = _construct_qgrid_1(xgrid, full=True) # (nx,)
        qshape = [qgrid.shape[0] if i==j else 1 for j in range(ndim+1)] # =(1,ny,1,1)
        newboxshape.append(qgrid.shape[0])
        qgrids.append(qgrid.view(*qshape))

    newboxshape = newboxshape + [2] # 2 for real and complex
    newboxshape = tuple(newboxshape)
    for i in range(ndim):
        qgrids[i] = qgrids[i].expand(*newboxshape).unsqueeze(-1) # (nx,ny,nz,2,1)

    qgrid = torch.cat(qgrids, dim=-1).view(-1,ndim) # (ns,ndim)
    return qgrid, newboxshape
