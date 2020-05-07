from abc import abstractmethod
import torch
import numpy as np
from numpy.polynomial.legendre import leggauss
from ddft.grids.base_grid import BaseGrid, BaseTransformed1DGrid
from ddft.utils.legendre import legint, legvander, legder, deriv_legval
from ddft.utils.interp import searchsorted

class LegendreRadialTransform(BaseTransformed1DGrid):
    def __init__(self, nx, dtype=torch.float, device=torch.device('cpu')):
        # cache variables
        self._spline_mat_inv_ = None

        xleggauss, wleggauss = leggauss(nx)
        self.xleggauss = torch.tensor(xleggauss, dtype=dtype, device=device)
        self.wleggauss = torch.tensor(wleggauss, dtype=dtype, device=device)
        self._boxshape = (nx,)

        self.rs = self.transform(self.xleggauss)
        self._rgrid = self.rs.unsqueeze(-1) # (nx, 1)

        # integration elements
        self._scaling = self.get_scaling(self.rs) # dr/dg
        self._dr = self._scaling * self.wleggauss
        self._dvolume = (4*np.pi*self.rs*self.rs) * self._dr

        # legendre basis (from tinydft/tinygrid.py)
        self.basis = legvander(self.xleggauss, nx-1, orderfirst=True) # (nr, nr)
        self.inv_basis = self.basis.inverse()

        # # construct the differentiation matrix
        # dlegval = deriv_legval(self.xleggauss, nx)
        # eye = torch.eye(nx, dtype=dtype, device=device)
        # dxleg = self.xleggauss - self.xleggauss.unsqueeze(-1) + eye
        # dmat = dlegval / (dlegval.unsqueeze(-1) * dxleg) # (nr, nr)
        # dmat_diag = self.xleggauss / (1. - self.xleggauss) / (1 + self.xleggauss) # (nr,)
        # self.diff_matrix = dmat * (1.-eye) + torch.diag_embed(dmat_diag)

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # the expression below is used to satisfy the following conditions:
        # * symmetric operator (by doing the integral 1/|r-r1|)
        # * 0 at r=\infinity, but not 0 at the bound (again, by doing the integral 1/|r-r1|)
        # to satisfy all the above, we choose to do the integral of
        #     Vlm(r) = integral_rmin^rmax (rless^l) / (rgreat^(l+1)) flm(r1) r1^2 dr1
        # where rless = min(r,r1) and rgreat = max(r,r1)

        # calculate the matrix rless / rgreat
        rless = torch.min(self.rs.unsqueeze(-1), self.rs) # (nr, nr)
        rgreat = torch.max(self.rs.unsqueeze(-1), self.rs)
        rratio = 1. / rgreat

        # the integralbox for radial grid is integral[4*pi*r^2 f(r) dr] while here
        # we only need to do integral[f(r) dr]. That's why it is divided by (4*np.pi)
        # and it is not multiplied with (self.radrgrid**2) in the lines below
        intgn = (f).unsqueeze(-2) * rratio # (nbatch, nr, nr)
        vrad_lm = self.integralbox(intgn / (4*np.pi), dim=-1)

        return -vrad_lm

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape

    @property
    def rgrid(self):
        return self._rgrid

    def interpolate(self, f, rq, extrap=None):
        # f: (nbatch, nr)
        # rq: (nrq, ndim)
        # return: (nbatch, nrq)
        nbatch, nr = f.shape
        nrq = rq.shape[0]

        rmax = self.rgrid.max()
        idxinterp = rq[:,0] <= rmax
        idxextrap = rq[:,0] > rmax
        allinterp = torch.all(idxinterp)
        if allinterp:
            rqinterp = rq[:,0]
        else:
            rqinterp = rq[idxinterp,0]

        # doing the interpolation
        # cubic interpolation is slower, but more robust on backward gradient
        xq = self.invtransform(rqinterp) # (nrq,)
        frqinterp = self._cubic_spline(xq, self.xleggauss, f) # cubic spline
        # coeff = torch.matmul(f, self.inv_basis) # (nbatch, nr)
        # basis = legvander(xq, nr-1, orderfirst=True)
        # frqinterp = torch.matmul(coeff, basis)

        if allinterp:
            return frqinterp

        # extrapolate
        if extrap is not None:
            frqextrap = extrap(rq[idxextrap,:])

        # combine the interpolation and extrapolation
        frq = torch.zeros((nbatch, nrq), dtype=rq.dtype, device=rq.device)
        frq[:,idxinterp] = frqinterp
        if extrap is not None:
            frq[:,idxextrap] = frqextrap

        return frq

    def grad(self, p, dim=-1, idim=0):
        if dim != -1:
            p = p.transpose(dim, -1) # (..., nr)

        # get the derivative w.r.t. the legendre basis
        coeff = torch.matmul(p, self.inv_basis) # (..., nr)
        dcoeff = legder(coeff) # (..., nr)
        dpdq = torch.matmul(dcoeff, self.basis) # (..., nr)
        # # multiply with the differentiation matrix to get dp/dq
        # dpdq = torch.matmul(p, self.diff_matrix)

        # get the derivative w.r.t. r
        dpdr = dpdq / self.get_scaling(self.rgrid[:,0])
        if dim != -1:
            dpdr = dpdr.transpose(dim, -1)
        return dpdr

    def laplace(self, p, dim=-1):
        if dim != -1:
            p = p.transpose(dim, -1) # p: (..., nr)

        pder1 = self.grad(p)
        pder2 = self.grad(pder1)
        res = pder2 + 2 * pder1 / (self.rs + 1e-15)

        if dim != -1:
            res = res.transpose(dim, -1)
        return res

    def _get_spline_mat_inverse(self):
        if self._spline_mat_inv_ is None:
            nx = self.xleggauss.shape[0]
            device = self.xleggauss.device
            dtype = self.xleggauss.dtype

            # construct the matrix for the left hand side
            dxinv0 = 1./(self.xleggauss[1:] - self.xleggauss[:-1]) # (nx-1,)
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
            self._spline_mat_inv_ = spline_mat_inv
        return self._spline_mat_inv_

    def _cubic_spline(self, xq, x, y):
        # https://en.wikipedia.org/wiki/Spline_interpolation#Algorithm_to_find_the_interpolating_cubic_spline
        # xq: (nrq,)
        # x: (nr,)
        # y: (nbatch, nr)

        # get the k-vector (i.e. the gradient at every points)
        spline_mat_inv = self._get_spline_mat_inverse()
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

class LegendreRadialShiftExp(LegendreRadialTransform):
    def __init__(self, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
        # setup the parameters needed for the transformation
        self.rmin = rmin
        self.logrmin = torch.tensor(np.log(rmin)).to(dtype).to(device)
        self.logrmax = torch.tensor(np.log(rmax)).to(dtype).to(device)
        self.logrmm = self.logrmax - self.logrmin

        super(LegendreRadialShiftExp, self).__init__(nr, dtype, device)

    def transform(self, xlg):
        return torch.exp((xlg + 1)*0.5 * self.logrmm + self.logrmin) - self.rmin

    def invtransform(self, rs):
        return (torch.log(rs + self.rmin) - self.logrmin) / (0.5 * self.logrmm) - 1.0

    def get_scaling(self, rs):
        return (rs + self.rmin) * self.logrmm * 0.5

    #################### editable module parts ####################
    def getparams(self, methodname):
        if methodname == "solve_poisson":
            return [self.rs, self._dvolume]
        elif methodname == "interpolate":
            return [self.logrmin, self.logrmm, self.xleggauss, self._spline_mat_inv_]
        elif methodname == "get_dvolume":
            return [self._dvolume]
        else:
            raise RuntimeError("The method %s has not been specified for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "solve_poisson":
            self.rs, self._dvolume = params[:2]
            return 2
        elif methodname == "interpolate":
            self.logrmin, self.logrmm, self.xleggauss, self._spline_mat_inv_ = params[:4]
            return 4
        elif methodname == "get_dvolume":
            self._dvolume, = params[:1]
            return 1
        else:
            raise RuntimeError("The method %s has not been specified for setparams" % methodname)

if __name__ == "__main__":
    import lintorch as lt
    grid = LegendreRadialShiftExp(1e-4, 1e2, 100, dtype=torch.float64)
    rgrid = grid.rgrid.clone().detach()
    f = torch.exp(-rgrid[:,0].unsqueeze(0)**2*0.5)

    lt.list_operating_params(grid.solve_poisson, f)
    lt.list_operating_params(grid.interpolate, f, rgrid)
    lt.list_operating_params(grid.get_dvolume)
