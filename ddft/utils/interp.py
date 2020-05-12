import numpy as np
import torch
from lintorch import EditableModule

def searchsorted(a, v, side="left"):
    idx = np.searchsorted(a.detach().numpy(), v.detach().numpy(), side=side)
    return torch.tensor(idx, dtype=torch.long)

class CubicSpline(EditableModule):
    def __init__(self, x):
        nx = x.shape[0]
        device = x.device
        dtype = x.dtype

        # construct the matrix for the left hand side
        dxinv0 = 1./(x[1:] - x[:-1]) # (nx-1,)
        dxinv = torch.cat((dxinv0[:1]*0, dxinv0, dxinv0[-1:]*0), dim=0)
        diag = (dxinv[:-1] + dxinv[1:]) * 2 # (nx,)
        offdiag = dxinv0 # (nx-1,)
        spline_mat = torch.zeros(nx, nx, dtype=dtype, device=device)
        spdiag = spline_mat.diagonal()
        spudiag = spline_mat.diagonal(offset=1)
        spldiag = spline_mat.diagonal(offset=-1)
        spdiag[:] = diag
        spudiag[:] = offdiag
        spldiag[:] = offdiag

        # construct the matrix on the right hand side
        dxinv2 = (dxinv * dxinv) * 3
        diagr = (dxinv2[:-1] - dxinv2[1:])
        udiagr = dxinv2[1:-1]
        ldiagr = -udiagr
        matr = torch.zeros(nx, nx, dtype=dtype, device=device)
        matrdiag = matr.diagonal()
        matrudiag = matr.diagonal(offset=1)
        matrldiag = matr.diagonal(offset=-1)
        matrdiag[:] = diagr
        matrudiag[:] = udiagr
        matrldiag[:] = ldiagr

        # solve the matrix inverse
        spline_mat_inv, _ = torch.solve(matr, spline_mat)
        self.spline_mat_inv = spline_mat_inv
        self.x = x

    def interp(self, y, xq):
        # https://en.wikipedia.org/wiki/Spline_interpolation#Algorithm_to_find_the_interpolating_cubic_spline
        # xq: (nrq,)
        # y: (nbatch, nr)

        # get the k-vector (i.e. the gradient at every points)
        x = self.x
        spline_mat_inv = self.spline_mat_inv
        ks = torch.matmul(y, spline_mat_inv.transpose(-2,-1)) # (nbatch, nr)

        # find the index location of xq
        nr = x.shape[0]
        idxr = searchsorted(x, xq)
        idxr = torch.clamp(idxr, 1, nr-1)
        idxl = idxr - 1 # (nrq,) from (0 to nr-2)

        if len(xq) > len(x):
            # get the variables needed
            yl = y[:,:-1]
            xl = x[:-1]
            dy = y[:,1:] - yl # (nbatch, nr-1)
            dx = x[1:] - xl # (nr-1)
            a = ks[:,:-1] * dx - dy # (nbatch, nr-1)
            b = -ks[:,1:] * dx + dy # (nbatch, nr-1)

            # calculate the coefficients for the t-polynomial
            p0 = yl # (nbatch, nr-1)
            p1 = (dy + a) # (nbatch, nr-1)
            p2 = (b - 2*a) # (nbatch, nr-1)
            p3 = a - b # (nbatch, nr-1)

            t = (xq - xl[idxl]) / (dx[idxl]) # (nrq,)
            yq = p0[:,idxl] + t * (p1[:,idxl] + t * (p2[:,idxl] + t * p3[:,idxl])) # (nbatch, nrq)
            return yq

        else:
            xl = x[idxl].contiguous()
            xr = x[idxr].contiguous()
            yl = y[:,idxl].contiguous()
            yr = y[:,idxr].contiguous()
            kl = ks[:,idxl].contiguous()
            kr = ks[:,idxr].contiguous()

            dxrl = xr - xl # (nrq,)
            dyrl = yr - yl # (nbatch, nrq)

            # calculate the coefficients of the large matrices
            t = (xq - xl) / dxrl # (nrq,)
            tinv = 1 - t # nrq
            tta = t*tinv*tinv
            ttb = t*tinv*t
            tyl = tinv + tta - ttb
            tyr = t - tta + ttb
            tkl = tta * dxrl
            tkr = -ttb * dxrl

            yq = yl*tyl + yr*tyr + kl*tkl + kr*tkr
            return yq

    def getparams(self, methodname):
        return [self.spline_mat_inv, self.x]

    def setparams(self, methodname, *params):
        self.spline_mat_inv, self.x = params[:2]
        return 2
