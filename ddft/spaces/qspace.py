import torch
import numpy as np
from ddft.spaces.base_space import BaseSpace

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
        self._qgrid, self._qboxshape = self._construct_qgrid(rgrid, boxshape)
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
        sigftbox = torch.rfft(sigbox, signal_ndim=self.ndim, onesided=True,
                normalized=True) # (...,nx,ny,nz//2+1,2)

        # for real fft, there are still redundancy even if we put onesided=True
        # for even n, the redundancies are in imag(X_0)=0 and imag(X_(N/2))=0
        # for odd n, the redundancy is in imag(X_0)=0
        sigftbox_eff = self._remove_redundancy(sigftbox) # (...,nx,ny,nz)

        sigft = self.flattensig(sigftbox_eff, dim=-1, qdom=True) # (...,ns)

        if transposed:
            sigft = sigft.transpose(dim,-1)

        return sigft # (...,ns,...)

    def invtransformsig(self, tsig, dim=-1):
        # tsig: (...,ns,...)
        # return: (...,nr,...)

        # normalize the dim (bring the dim to -1)
        tsig, transposed = _normalize_dim(tsig, dim, -1)

        tsigbox = self.boxifysig(tsig, dim=-1, qdom=True) #(...,nx,ny,nz)
        # add the redundancy
        tsigbox = self._add_redundancy(tsigbox)
        sigbox = torch.irfft(tsigbox, signal_ndim=self.ndim, onesided=True,
                normalized=True, signal_sizes=self.boxshape) # (...,nx,ny,nz)
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
        sig = self.invtransformsig(tsig, -1)

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

    def _construct_qgrid_1(self, xgrid, full=False):
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

    def _construct_qgrid(self, rgrid, boxshape):
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
            qgrid = self._construct_qgrid_1(xgrid, full=(i<ndim-1)) # (nx,)
            qshape = [qgrid.shape[0] if i==j else 1 for j in range(ndim+1)] # =(1,ny,1,1)
            newboxshape.append(qgrid.shape[0])
            qgrids.append(qgrid.view(*qshape))

        newboxshape = newboxshape + [2] # 2 for cosine and sine factor (or real & imag)
        newboxshape = tuple(newboxshape)
        for i in range(ndim):
            qgridsi = qgrids[i].expand(*newboxshape).contiguous() # (nx,ny,nz//2+1,2)
            # remove the redundancy
            qgrids_eff = self._remove_redundancy(qgridsi) # (nx,ny,nz)

            qgrids[i] = qgrids_eff.unsqueeze(-1) # (nx,ny,nz,1)

        qgrid = torch.cat(qgrids, dim=-1).view(-1,ndim) # (ns,ndim)
        return qgrid, boxshape

    def _remove_redundancy(self, sig):
        # sig: (...,nz//2+1, 2)
        sigflat = sig.view(*sig.shape[:-2], -1) # (...,(nz//2)*2+2)
        nlast = sigflat.shape[-1]

        # check the index
        if not hasattr(self, "redundancy_index"):
            if self.is_even:
                self.redundancy_index = torch.LongTensor([i for i in range(nlast) if (i!=1 and i!=(nlast-1))])
            else:
                self.redundancy_index = torch.LongTensor([i for i in range(nlast) if (i!=1)])
        sig_eff = sigflat.index_select(dim=-1, index=self.redundancy_index) # (...,nz)
        return sig_eff

    def _add_redundancy(self, tsigbox):
        # tsigbox: (...,nx,ny,nz)
        tsigbox_flat = tsigbox.view(-1,tsigbox.shape[-1]) # (...*nx*ny,nz)
        redundancy = torch.zeros(tsigbox_flat.shape[0], 1).to(tsigbox.dtype).to(tsigbox.device)
        if self.is_even:
            tsigbox_flat = torch.cat((tsigbox_flat[:,:1], redundancy, tsigbox_flat[:,1:-1], redundancy), dim=-1) # (...*nx*ny,nz+2)
        else:
            tsigbox_flat = torch.cat((tsigbox_flat[:,:1], redundancy, tsigbox_flat[:,1:]), dim=-1) # (...*nx*ny,nz+1)
        tsigbox = tsigbox_flat.view(*tsigbox.shape[:-1], -1, 2) # (...,nx,ny,nz//2+1,2)
        return tsigbox # (...,nx,ny,nz//2+1,2)

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
