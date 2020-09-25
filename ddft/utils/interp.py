import numpy as np
import torch
from xitorch import EditableModule

# @torch.jit.script
def get_spline_mat_inv(x:torch.Tensor, transpose:bool=True):
    """
    Returns the inverse of spline matrix where the gradient can be obtained just
    by

    >>> spline_mat_inv = get_spline_mat_inv(x, transpose=True)
    >>> ks = torch.matmul(y, spline_mat_inv)

    where `y` is a tensor of (nbatch, nr) and `spline_mat_inv` is the output of
    this function with shape (nr, nr)

    Arguments
    ---------
    * x: torch.Tensor with shape (nb, nr)
        The x-position of the data
    * transpose: bool
        If true, then transpose the result.

    Returns
    -------
    * mat: torch.Tensor with shape (nb, nr, nr)
        The inverse of spline matrix.
    """
    nb, nr = x.shape

    device = x.device
    dtype = x.dtype

    # construct the matrix for the left hand side
    dxinv0 = 1./(x[:,1:] - x[:,:-1]) # (nb,nr-1)
    dxinv = torch.cat((dxinv0[:,:1]*0, dxinv0, dxinv0[:,-1:]*0), dim=-1)
    diag = (dxinv[:,:-1] + dxinv[:,1:]) * 2 # (nb,nr)
    offdiag = dxinv0 # (nb,nr-1)
    spline_mat = torch.zeros(nb, nr, nr, dtype=dtype, device=device)
    spdiag = spline_mat.diagonal(dim1=-2, dim2=-1)
    spudiag = spline_mat.diagonal(offset=1, dim1=-2, dim2=-1)
    spldiag = spline_mat.diagonal(offset=-1, dim1=-2, dim2=-1)
    spdiag[:,:] = diag
    spudiag[:,:] = offdiag
    spldiag[:,:] = offdiag

    # construct the matrix on the right hand side
    dxinv2 = (dxinv * dxinv) * 3
    diagr = (dxinv2[:,:-1] - dxinv2[:,1:])
    udiagr = dxinv2[:,1:-1]
    ldiagr = -udiagr
    matr = torch.zeros(nb, nr, nr, dtype=dtype, device=device)
    matrdiag = matr.diagonal(dim1=-2, dim2=-1)
    matrudiag = matr.diagonal(offset=1, dim1=-2, dim2=-1)
    matrldiag = matr.diagonal(offset=-1, dim1=-2, dim2=-1)
    matrdiag[:,:] = diagr
    matrudiag[:,:] = udiagr
    matrldiag[:,:] = ldiagr

    # solve the matrix inverse
    spline_mat_inv, _ = torch.solve(matr, spline_mat)
    if transpose:
        spline_mat_inv = spline_mat_inv.transpose(-2,-1)

    # return to the shape of x
    return spline_mat_inv
