import torch
import numpy as np
from ddft.spaces.base_space import BaseSpace
from ddft.maths.rfft import rfft, irfft

class QSpace(BaseSpace):
    """
    Q-space is the Cosine and Sine space for real signal in the spatial domain.
    It transforms real-valued signal in spatial domain to real-valued signal
        in the Cosine-Sine domain.
    """
    def __init__(self, rgrid, boxshape):
        # rgrid is (nr,ndim)
        # boxshape is (nx,ny,nz) for 3D, (nx,ny) for 2D, and (nx,) for 1D
        # qgrid is (ns, ndim)
        # qboxshape is (nx,ny,nz//2,2)
        self._rgrid = rgrid
        self._boxshape = boxshape
        self.is_even = (boxshape[-1]%2 == 0)
        self._qgrid, self._qboxshape = _construct_qgrid(rgrid, boxshape)
        self.nr = self._rgrid.shape[0]

        # halving signal for H transforms and inv H transforms
        self.halv = torch.ones(self._qgrid.shape[0]).to(rgrid.dtype).to(rgrid.device)
        if self.is_even:
            self.halv[2:-1] = 0.5
        else:
            self.halv[2:] = 0.5

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

        # normalize the dim (bring the dim to -1)
        sig, transposed = _normalize_dim(sig, dim, -1)

        sigbox = self.boxifysig(sig, dim=-1) # (...,...,nx,ny,nz)
        sigftbox = rfft(sigbox, signal_ndim=self.ndim, normalized=True) # (...,nx,ny,nz)
        sigft = self.flattensig(sigftbox, dim=-1, qdom=True) # (...,ns)

        if transposed:
            sigft = sigft.transpose(dim,-1)

        return sigft # (...,ns,...)

    def invtransformsig(self, tsig, dim=-1):
        # tsig: (...,ns,...)
        # return: (...,nr,...)

        # normalize the dim (bring the dim to -1)
        tsig, transposed = _normalize_dim(tsig, dim, -1)

        tsigbox = self.boxifysig(tsig, dim=-1, qdom=True) #(...,nx,ny,nz)
        sigbox = irfft(tsigbox, signal_ndim=self.ndim, normalized=True) # (...,nx,ny,nz)
        sig = self.flattensig(sigbox, dim=-1) # (...,nr)

        if transposed:
            sig = sig.transpose(dim,-1)
        return sig # (...,nr,...)

    def Ttransformsig(self, tsig, dim=-1):
        # the Ttransformsig of qspace is equal to invtransformsig, but with the
        # non-redundant input signal is halved.
        # tsig : (...,ns,...)
        # return: (...,nr,...)

        # bring the dim to -1
        tsig, transposed = _normalize_dim(tsig, dim, -1) # tsig: (...,ns)

        # halving the middle input signal to invtransformsig
        tsig_half = tsig * self.halv
        sig = self.invtransformsig(tsig_half, -1)

        if transposed:
            sig = sig.transpose(dim, -1)
        return sig

    def invTtransformsig(self, sig, dim=-1):
        # the invTtransformsig of qspace is equal to transformsig, but double the
        # middle of output signal

        # bring the dim to -1
        sig, transposed = _normalize_dim(sig, dim, -1)

        # double the output signal
        tsig_half = self.transformsig(sig, -1)
        tsig = tsig_half / self.halv

        if transposed:
            tsig = tsig.transpose(dim, -1)
        return tsig

    @property
    def isorthogonal(self):
        return False

def _construct_qgrid_1(xgrid):
    N = len(xgrid)
    dr = xgrid[1] - xgrid[0]
    boxsize = xgrid[-1] - xgrid[0]
    dq = 2*np.pi / (N * dr)
    Nhalf = (N // 2) + 1
    offset = (N + 1) % 2
    qgrid = torch.arange(Nhalf)
    qgrid = qgrid.to(xgrid.dtype).to(xgrid.device) * dq
    return qgrid

def _construct_qgrid(rgrid, boxshape):
    # rgrid: (nr, ndim)
    # boxshape = (nx, ny, nz) for 3D

    nr, ndim = rgrid.shape
    rgrid = rgrid.view(*boxshape, ndim) # (nx, ny, nz, ndim)
    qgrids = []
    spacing = nr * ndim
    for i in range(ndim):
        spacing = spacing // boxshape[i]
        na = rgrid.shape[i]
        index = torch.arange(rgrid.shape[i]) * spacing + i
        xgrid = torch.take(rgrid, index=index)
        qgrid = _construct_qgrid_1(xgrid) # (nx//2+1,)
        qgrid_double = torch.repeat_interleave(qgrid, 2, dim=-1) # ((nx//2)*2+2)

        # remove the redundancy
        if na%2 == 0: # is even
            qgrideff = qgrid_double[1:-1]
        else: # odd
            qgrideff = qgrid_double[1:] # (nx,)
        qshape = [qgrideff.shape[0] if i==j else 1 for j in range(ndim)] # =(1,ny,1,1)
        qgrids.append(qgrideff.view(*qshape).expand(*boxshape).unsqueeze(-1)) # (nx,ny,nz,1)

    qgrid = torch.cat(qgrids, dim=-1).view(-1,ndim)
    return qgrid, boxshape

def _normalize_dim(tsig, dim, dest_dim=-1):
    ndim = tsig.ndim
    if dim < 0:
        dim = ndim + dim
    if dest_dim < 0:
        dest_dim = ndim + dest_dim

    # put the nr at dim=dest_dim
    transposed = (dim != dest_dim)
    if transposed:
        tsig = tsig.transpose(dim, dest_dim) # (...,ns)
    return tsig, transposed
