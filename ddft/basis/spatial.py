from abc import abstractmethod
import torch
import lintorch as lt

class SpatialBasis(object):
    def __init__(self, rgrid, boxshape):
        # rgrid: (nr, ndim)
        # boxshape: (ndim,)
        self._rgrid = rgrid
        self._boxshape = boxshape
        self._is_even = (boxshape[-1]%2 == 0)
        # qgrid is 
        self._qgrid, self._qboxshape = _construct_qgrid(rgrid, boxshape)
        self._nr = self._rgrid.shape[0]

    def nbasis(self):
        return self._nr

    @abstractmethod
    def kinetics(self):
        """
        Returns a lintorch.Module for the Kinetics operator.
        """
        pass

    @abstractmethod
    def vcoulomb(self):
        """
        Returns a lintorch.Module for the Coulomb potential operator with Z=1.
        """
        pass

    @abstractmethod
    def vpot(self, vext):
        """
        Returns a lintorch.Module for the given external potential operator.

        Arguments
        ---------
        * vext: torch.tensor (nbatch, nr)
        """
        pass

    @abstractmethod
    def overlap(self):
        """
        Returns the basis overlap operator in lintorch.Module.
        If the basis are orthogonal, then it should return an identity operator.
        """
        pass

    def tocoeff(self, wfr):
        return wfr

    def frocoeff(self, coeffs):
        return coeffs

def _construct_qgrid_1(xgrid):
    N = len(xgrid)
    dr = xgrid[1] - xgrid[0]
    boxsize = xgrid[-1] - xgrid[0]
    dq = 2*np.pi / (N * dr)
    Nhalf = (N // 2) + 1
    offset = (N + 1) % 2
    qgrid_left = torch.arange(Nhalf)
    qgrid_right = -torch.arange(Nhalf-offset-1,0,-1)
    qgrid = torch.cat((qgrid_left, qgrid_right))
    qgrid = qgrid.to(xgrid.dtype).to(xgrid.device) * dq
    return qgrid

def _construct_qgrid(rgrid, boxshape):
    # rgrid: (nr, ndim)
    # boxshape = (nx, ny, nz) for 3D

    nr, ndim = rgrid.shape
    rgrid = rgrid.view(*boxshape, ndim) # (nx, ny, nz, ndim)
    qgrids = []
    spacing = nr * ndim
    newboxshape = [*boxshape, 2]
    for i in range(ndim):
        spacing = spacing // boxshape[i]
        na = rgrid.shape[i]
        index = torch.arange(rgrid.shape[i]) * spacing + i
        index = index.to(rgrid.device)
        xgrid = torch.take(rgrid, index=index)
        qgrid = _construct_qgrid_1(xgrid) # (nx,)
        qshape = [qgrid.shape[0] if i==j else 1 for j in range(ndim+1)] # =(1,ny,1,1)
        qgrids.append(qgrid.view(*qshape).expand(*newboxshape).unsqueeze(-1)) # (nx,ny,nz,2,1)

    qgrid = torch.cat(qgrids, dim=-1)
    return qgrid.view(-1, ndim), newboxshape

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
