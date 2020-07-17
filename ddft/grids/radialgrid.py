from abc import abstractmethod, abstractproperty
import torch
import numpy as np
from numpy.polynomial.legendre import leggauss
from ddft.grids.base_grid import BaseGrid, BaseTransformed1DGrid
from ddft.utils.legendre import legint, legvander, legder, deriv_legval
from ddft.utils.interp import CubicSpline
from ddft.grids.radialtransform import ShiftExp, DoubleExp2

class RadialGrid(BaseGrid):
    @abstractproperty
    def interpolator(self):
        pass

    @abstractproperty
    def transformobj(self):
        pass

    @abstractmethod
    def get_dvolume(self):
        pass

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # the expression below is used to satisfy the following conditions:
        # * symmetric operator (by doing the integral 1/|r-r1|)
        # * 0 at r=\infinity, but not 0 at the bound (again, by doing the integral 1/|r-r1|)
        # to satisfy all the above, we choose to do the integral of
        #     Vlm(r) = integral_rmin^rmax (rless^l) / (rgreat^(l+1)) flm(r1) r1^2 dr1
        # where rless = min(r,r1) and rgreat = max(r,r1)

        # calculate the matrix rless / rgreat
        rs = self.rgrid[:,0]
        rless = torch.min(rs.unsqueeze(-1), rs) # (nr, nr)
        rgreat = torch.max(rs.unsqueeze(-1), rs)
        rratio = 1. / rgreat

        # the integralbox for radial grid is integral[4*pi*r^2 f(r) dr] while here
        # we only need to do integral[f(r) dr]. That's why it is divided by (4*np.pi)
        # and it is not multiplied with (self.radrgrid**2) in the lines below
        intgn = (f).unsqueeze(-2) * rratio # (nbatch, nr, nr)
        vrad_lm = self.integralbox(intgn / (4*np.pi), dim=-1)

        return -vrad_lm

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
        xq = self.transformobj.invtransform(rqinterp) # (nrq,)
        frqinterp = self.interpolator.interp(f, xq)
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

    @abstractmethod
    def grad(self, p, dim, idim):
        pass

    def laplace(self, p, dim=-1):
        if dim != -1:
            p = p.transpose(dim, -1) # p: (..., nr)

        pder1 = self.grad(p)
        pder2 = self.grad(pder1)
        res = pder2 + 2 * pder1 / (self.rgrid[:,0] + 1e-15)

        if dim != -1:
            res = res.transpose(dim, -1)
        return res

    def getparams(self, methodname):
        if methodname == "solve_poisson":
            return self.getparams("get_dvolume")
        elif methodname == "interpolate":
            return self.transformobj.getparams("invtransform") + \
                   self.interpolator.getparams("interp")
        elif methodname == "laplace":
            return self.getparams("laplace")
        else:
            raise RuntimeError("Unimplemented %s for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "solve_poisson":
            return self.setparams("get_dvolume", *params)
        elif methodname == "interpolate":
            idx = 0
            idx += self.transformobj.setparams("invtransform", *params)
            idx += self.interpolator.setparams("interp", *params[idx:])
            return idx
        elif methodname == "laplace":
            return self.setparams("laplace", *params)
        else:
            raise RuntimeError("Unimplemented %s for setparams" % methodname)

class M1P1TransformRadialGrid(RadialGrid):
    """
    Grid with defined point in the range [0,inf) by transformation from [-1,1].
    """
    def __init__(self, x, w, transformobj, dtype=torch.float, device=torch.device('cpu')):
        self.x = x
        self.w = w
        self._boxshape = (len(x),)
        self._interpolator = CubicSpline(self.x)

        self._transformobj = transformobj
        if not isinstance(transformobj, BaseTransformed1DGrid):
            raise TypeError("transformobj must be BaseTransformed1DGrid")
        self.rs = self.transformobj.transform(self.x)
        self._rgrid = self.rs.unsqueeze(-1) # (nx, 1)

        # integration elements
        self._scaling = self.transformobj.get_scaling(self.rs) # dr/dg
        self._dr = self._scaling * self.w
        self._dvolume = (4*np.pi*self.rs*self.rs) * self._dr

    @property
    def interpolator(self):
        return self._interpolator

    @property
    def transformobj(self):
        return self._transformobj

    def get_dvolume(self):
        return self._dvolume

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape

    @abstractmethod
    def grad(self, p, dim=-1, idim=0):
        pass

    def getparams(self, methodname):
        if methodname == "get_dvolume":
            return [self._dvolume]
        else:
            return super().getparams(methodname)

    def setparams(self, methodname, *params):
        if methodname == "get_dvolume":
            self._dvolume, = params[:1]
            return 1
        else:
            return super().setparams(methodname, *params)

class NaiveRadialGrid(M1P1TransformRadialGrid):
    def __init__(self, nx, transformobj, dtype=torch.float, device=torch.device('cpu')):
        x = torch.linspace(-1, 1, nx)
        w = self.x[1] - self.x[0]
        super(NaiveRadialGrid, self).__init__(x=x, w=w,
            transformobj=transformobj, dtype=dtype, device=device)

    def grad(self, p, dim=-1, idim=0):
        if dim != -1:
            p = p.transpose(dim, -1) # (..., nr)

        dpdq = torch.cat((
            p[:,1:2] - p[:,0:1],
            (p[:,2:] - p[:,:-2])*0.5,
            p[:,-1].unsqueeze(-1) - p[:,-2:-1],
        ), dim=-1) # (..., nr)

        # get the derivative w.r.t. r
        dpdr = dpdq / self.transformobj.get_scaling(self.rgrid[:,0])
        if dim != -1:
            dpdr = dpdr.transpose(dim, -1)
        return dpdr

class GaussChebyshevRadialGrid(M1P1TransformRadialGrid):
    def __init__(self, n, transformobj, dtype=torch.float, device=torch.device('cpu')):
        # generate the x and w from chebyshev polynomial
        np1 = n+1.
        ipn1 = np.arange(n,0,-1) * np.pi / np1
        sin_ipn1 = np.sin(ipn1)
        sin_ipn1_2 = sin_ipn1 * sin_ipn1
        xcheb = (np1-2*i) / np1 + 2/np.pi * (1 + 2./3 * sin_ipn1*sin_ipn1) * np.cos(ipn1) * sin_ipn1
        wcheb = 16. / (3*np1) * sin_ipn1_2 * sin_ipn1_2

        xcheb = torch.tensor(xcheb, dtype=dtype, device=device)
        wcheb = torch.tensor(wcheb, dtype=dtype, device=device)
        super(GaussChebyshevRadialGrid, self).__init__(x=xcheb, w=wcheb,
            transformobj=transformobj, dtype=dtype, device=device)

    def grad(self, p, dim=-1, idim=0):
        pass # ???

class LegendreRadialGrid(M1P1TransformRadialGrid):
    def __init__(self, nx, transformobj, dtype=torch.float, device=torch.device('cpu')):
        xleggauss, wleggauss = leggauss(nx)
        self.xleggauss = torch.tensor(xleggauss, dtype=dtype, device=device)
        self.wleggauss = torch.tensor(wleggauss, dtype=dtype, device=device)
        super(LegendreRadialGrid, self).__init__(x=self.xleggauss, w=self.wleggauss,
            transformobj=transformobj, dtype=dtype, device=device)

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
        dpdr = dpdq / self.transformobj.get_scaling(self.rgrid[:,0])
        if dim != -1:
            dpdr = dpdr.transpose(dim, -1)
        return dpdr

class LegendreRadialShiftExp(LegendreRadialGrid):
    def __init__(self, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
        # setup the parameters needed for the transformation
        transformobj = ShiftExp(rmin, rmax, dtype=dtype, device=device)
        super(LegendreRadialShiftExp, self).__init__(nr, transformobj, dtype=dtype, device=device)

class LegendreRadialDoubleExp2(LegendreRadialGrid):
    def __init__(self, alpha, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
        # setup the parameters needed for the transformation
        transformobj = DoubleExp2(alpha, rmin, rmax, dtype=dtype, device=device)
        super(LegendreRadialDoubleExp2, self).__init__(nr, transformobj, dtype=dtype, device=device)

if __name__ == "__main__":
    import lintorch as lt
    grid = LegendreRadialShiftExp(1e-4, 1e2, 100, dtype=torch.float64)
    rgrid = grid.rgrid.clone().detach()
    f = torch.exp(-rgrid[:,0].unsqueeze(0)**2*0.5)

    lt.list_operating_params(grid.solve_poisson, f)
    lt.list_operating_params(grid.interpolate, f, rgrid)
    lt.list_operating_params(grid.get_dvolume)
