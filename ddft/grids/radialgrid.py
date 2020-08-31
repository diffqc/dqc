from abc import abstractmethod, abstractproperty
import torch
import numpy as np
import lintorch as lt
import lintorch.optimize
from numpy.polynomial.legendre import leggauss
from ddft.grids.base_grid import BaseGrid
from ddft.utils.legendre import legint, legvander, legder, deriv_legval
from ddft.utils.interp import CubicSpline
from ddft.utils.cumsum_quad import CumSumQuad
from ddft.utils.safeops import eps as util_eps

__all__ = ["RadialGrid"]

class RadialGrid(BaseGrid):
    """
    Radial grid typically consists of a fixed interval grid and a transformation
    to transform from the interval of the fixed interval grid to [0,inf) for
    radial value.

    Arguments
    ---------
    * fixintvgrid: BaseFixedIntervalGrid or str
        Object of fixed interval integrator grid or string indicating the class.
        If this is a string, then `gridkwargs` is used as the kwargs of the class.
    * transformobj: BaseGridTransformation or str or None
        Transformation object or a string indicating the class.
        The transformation object should transform the initial range of the grid
        into (0,inf] as the radial range.
        If this is a string, then `tfmkwargs` is used as the kwargs of the class.
        If None, then the "identity" transformation is used.
    * gridkwargs: dict
        The kwargs of the grid object to be used to construct the fixed interval
        integrator grid object. Only used if `fixintvgrid` is a string.
    * tfmkwargs: dict
        The kwargs of the transformation object to be used to construct the
        range transformation object. Only used if `transformobj` is a string.
    """
    def __init__(self, fixintvgrid, transformobj=None, gridkwargs={}, tfmkwargs={}):
        # get the grid object
        if isinstance(fixintvgrid, BaseFixedIntervalGrid):
            grid = fixintvgrid
        elif isinstance(fixintvgrid, str):
            grid = get_fixed_interval_grid(fixintvgrid)(**gridkwargs)
        else:
            raise TypeError("Argument fixintvgrid must be BaseFixedIntervalGrid or a string")

        # get the transformation object
        if isinstance(transformobj, BaseGridTransformation):
            tfmobj = transformobj
        elif isinstance(transformobj, str):
            tfmobj = get_transformation(transformobj)(**tfmkwargs)
        elif transformobj is None:
            tfmobj = get_transformation("identity")()
        else:
            raise TypeError("Argument transformobj must be BaseGridTransformation or a string")

        self.x, self.w = grid.get_xw()
        self.z, self.dxdz = grid.get_z_dxdz() # (nx,)
        self._boxshape = (len(self.x),)
        self._interpolator = CubicSpline(self.x)
        self._cumsumquad = CumSumQuad(self.z, side="both", method="simpson")

        self._transformobj = tfmobj
        self.rs = self.transformobj.transform(self.x) # (nx,)
        self._rgrid = self.rs.unsqueeze(-1) # (nx, 1)

        # integration elements
        self._scaling = self.transformobj.get_scaling(self.rs) # dr/dg # (nx,)
        self._vol_elmt = 4*np.pi*self.rs*self.rs # (nx,)
        self._dr = self._scaling * self.w
        self._dvolume = self._vol_elmt * self._dr

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

    def grad(self, p, dim=-1, idim=0):
        pass # ???

    def laplace(self, p, dim=-1):
        pass # ???

    def cumsum_integrate(self, f): # batchified
        # f: (*BF, nr, nr) in r-space
        dvol = (self._scaling * self._vol_elmt * self.dxdz).unsqueeze(-2) # (1, nr)
        fx = f * dvol # (*BF, nr, nr)
        return self._cumsumquad.integrate(fx) # (nbatch, nr)

    def solve_poisson(self, f): # batchified
        # f: (*BF, nr)
        # the expression below is used to satisfy the following conditions:
        # * symmetric operator (by doing the integral 1/|r-r1|)
        # * 0 at r=\infinity, but not 0 at the bound (again, by doing the integral 1/|r-r1|)
        # to satisfy all the above, we choose to do the integral of
        #     Vlm(r) = integral_rmin^rmax (rless^l) / (rgreat^(l+1)) flm(r1) r1^2 dr1
        # where rless = min(r,r1) and rgreat = max(r,r1)

        # calculate the matrix rless / rgreat
        rs = self.rgrid[:,0] # (nr,)
        rless  = torch.min(rs.unsqueeze(-1), rs.unsqueeze(-2)) # (nr, nr)
        rgreat = torch.max(rs.unsqueeze(-1), rs.unsqueeze(-2))
        rratio = 1. / rgreat

        # the integralbox for radial grid is integral[4*pi*r^2 f(r) dr] while here
        # we only need to do integral[f(r) dr]. That's why it is divided by (4*np.pi)
        # and it is not multiplied with (self.radrgrid**2) in the lines below
        intgn = (f).unsqueeze(-2) * rratio # (*BF, nr, nr)
        vrad_lm = self.cumsum_integrate(intgn) / (4*np.pi) # (*BF, nr)

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

    def getparamnames(self, methodname, prefix=""):
        if methodname == "get_dvolume":
            return [prefix+"_dvolume"]
        elif methodname == "solve_poisson":
            return [prefix+"rgrid"] + self.getparams("cumsum_integrate", prefix=prefix)
        elif methodname == "cumsum_integrate":
            return [prefix+"_scaling", prefix+"_vol_elmt", prefix+"dxdz"] + \
                   self._cumsumquad.getparamnames("integrate", prefix=prefix+"_cumsumquad.")
        elif methodname == "interpolate":
            return self.transformobj.getparamnames("invtransform", prefix=prefix+"transformobj.") + \
                   self.interpolator.getparamnames("interp", prefix=prefix+"interpolator.")
        else:
            return super().getparamnames(methodname, prefix=prefix)


def get_fixed_interval_grid(gridstr):
    s = gridstr.lower().replace("-", "")
    if s == "gausschebyshev" or s == "chebyshev":
        return GaussChebyshevGrid
    elif s == "legendre":
        return LegendreGrid
    else:
        raise RuntimeError("Unknown fixed interval grid: %s" % gridstr)

def get_transformation(tfmstr):
    s = tfmstr.lower().replace("-", "")
    if s == "identity":
        return IdentityTransformation
    elif s == "logm3":
        return LogM3Transformation
    elif s == "shiftexp":
        return ShiftExpTransformation
    else:
        raise RuntimeError("Unknown grid transformation: %s" % tfmstr)

########################### aliases for common grids ###########################
class LegendreShiftExpRadGrid(RadialGrid):
    def __init__(self, nx, rmin, rmax, dtype=torch.float, device=torch.device('cpu')):
        grid = LegendreGrid(nx, dtype=dtype, device=device)
        tfm = ShiftExpTransformation(rmin, rmax, dtype=dtype, device=device)
        super(LegendreShiftExpRadGrid, self).__init__(grid, tfm)

class LegendreDoubleExp2RadGrid(RadialGrid):
    def __init__(self, nr, alpha, rmin, rmax, dtype=torch.float, device=torch.device('cpu')):
        grid = LegendreGrid(nr, dtype=dtype, device=device)
        tfm = DoubleExp2Transformation(alpha, rmin, rmax, dtype=dtype, device=device)
        super(LegendreDoubleExp2RadGrid, self).__init__(grid, tfm)

class LegendreLogM3RadGrid(RadialGrid):
    def __init__(self, nr, ra=1.0, eps=None, dtype=torch.float, device=torch.device('cpu')):
        grid = LegendreGrid(nr, dtype=dtype, device=device)
        tfm = LogM3Transformation(ra, eps=eps, dtype=dtype, device=device)
        super(LegendreLogM3RadGrid, self).__init__(grid, tfm)

class GaussChebyshevLogM3RadGrid(RadialGrid):
    def __init__(self, nr, ra=1.0, eps=None, dtype=torch.float, device=torch.device('cpu')):
        grid = GaussChebyshevGrid(nr, dtype=dtype, device=device)
        tfm = LogM3Transformation(ra, eps=eps, dtype=dtype, device=device)
        super(GaussChebyshevLogM3RadGrid, self).__init__(grid, tfm)

############################# fixed interval grid #############################
class BaseFixedIntervalGrid(object):
    @abstractmethod
    def get_xw(self):
        """
        Get the x and weights, w, for the integration. x is the points from -1
        to 1 of the integration points and w is the integration weights.

        Returns
        -------
        * x: torch.tensor (nx,)
            The integration points.
        * w: torch.tensor (nx,)
            The integration weights.
        """
        pass

    def get_z_dxdz(self):
        """
        Returns the almost-uniformly spaced points, z, and the scaling dx/dz.
        The almost-uniformly spaced points are useful in cumulative integral
        using Simpson's rule or other quadrature rules.

        Returns
        -------
        * z: torch.tensor (nx,)
            The almost-uniformly spaced points.
        * dxdz: torch.tensor (nx,)
            The scaling dx/dz.
        """
        # the default is z=x and dxdz = 1
        z, _ = self.get_xw()
        dxdz = torch.ones_like(z)
        return z, dxdz

    @abstractmethod
    def grad(self, p, dim=-1, idim=0):
        pass # ???

    @abstractmethod
    def laplace(self, p, dim=-1):
        pass # ???

class GaussChebyshevGrid(BaseFixedIntervalGrid):
    def __init__(self, n, dtype=torch.float, device=torch.device('cpu')):
        # generate the x and w from chebyshev polynomial
        np1 = n+1.
        icount = np.arange(n,0,-1)
        ipn1 = icount * np.pi / np1
        sin_ipn1 = np.sin(ipn1)
        sin_ipn1_2 = sin_ipn1 * sin_ipn1
        xcheb = (np1-2*icount) / np1 + 2/np.pi * (1 + 2./3 * sin_ipn1*sin_ipn1) * np.cos(ipn1) * sin_ipn1
        wcheb = 16. / (3*np1) * sin_ipn1_2 * sin_ipn1_2

        self.xcheb = torch.tensor(xcheb, dtype=dtype, device=device)
        self.wcheb = torch.tensor(wcheb, dtype=dtype, device=device)
        self.z = torch.tensor(icount, dtype=dtype, device=device)
        self.dxdz = -self.wcheb

    def get_xw(self):
        return self.xcheb, self.wcheb

    def get_z_dxdz(self):
        return self.z, self.dxdz

    def grad(self, p, dim=-1, idim=0):
        pass # ???

    def laplace(self, p, dim=-1):
        pass # ???

class LegendreGrid(BaseFixedIntervalGrid):
    def __init__(self, nx, dtype=torch.float, device=torch.device('cpu')):
        xleggauss, wleggauss = leggauss(nx)
        self.xleggauss = torch.tensor(xleggauss, dtype=dtype, device=device)
        self.wleggauss = torch.tensor(wleggauss, dtype=dtype, device=device)

        # legendre basis (from tinydft/tinygrid.py)
        self.basis = legvander(self.xleggauss, nx-1, orderfirst=True) # (nr, nr)
        self.inv_basis = self.basis.inverse()

    def get_xw(self):
        return self.xleggauss, self.wleggauss

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

    def laplace(self, p, dim=-1):
        pass # ???

############################# grid transformation #############################
class BaseGridTransformation(object):
    @abstractmethod
    def transform(self, xlg):
        """
        Transform the coordinate from [-1,1] to the intended coordinate.
        """
        pass

    @abstractmethod
    def invtransform(self, rs):
        """
        Transform back from the intended coordinate to the coordinate [-1,1].
        """
        pass

    @abstractmethod
    def get_scaling(self, rs):
        """
        Obtain the scaling dr/dx for the integration.
        """
        pass

class IdentityTransformation(BaseGridTransformation):
    def transform(self, xlg):
        return xlg

    def invtransform(self, rs):
        return rs

    def get_scaling(self, rs):
        return torch.ones_like(rs).to(rs.device)

    def getparamnames(self, methodname, prefix=""):
        return []


class LogM3Transformation(BaseGridTransformation):
    # eq (12) in https://aip.scitation.org/doi/pdf/10.1063/1.475719
    def __init__(self, ra=1.0, eps=None, dtype=torch.float, device=torch.device('cpu')):
        # setup the parameters needed for the transformation
        if not isinstance(ra, torch.Tensor):
            ra = torch.tensor(ra, dtype=dtype, device=device)
        self.ra = ra
        self.eps = eps if eps is not None else util_eps
        self.ln2 = np.log(2.0 + self.eps)

    def transform(self, xlg):
        # eps to safeguard from instability
        return self.ra * (1 - torch.log1p(-xlg + self.eps) / self.ln2)

    def invtransform(self, rs):
        return -torch.expm1(self.ln2 * (1. - rs/self.ra)) + self.eps

    def get_scaling(self, rs):
        return self.ra / self.ln2 * torch.exp(-self.ln2 * (1. - rs / self.ra))

    #################### editable module parts ####################
    def getparamnames(self, methodname, prefix=""):
        if methodname == "invtransform" or methodname == "transform" or methodname == "get_scaling":
            return [prefix+"ra"]
        else:
            return super().getparamnames(methodname, prefix=prefix)


class ShiftExpTransformation(BaseGridTransformation):
    def __init__(self, rmin, rmax, dtype=torch.float, device=torch.device('cpu')):
        # setup the parameters needed for the transformation
        self.rmin = rmin
        self.logrmin = torch.tensor(np.log(rmin)).to(dtype).to(device)
        self.logrmax = torch.tensor(np.log(rmax)).to(dtype).to(device)
        self.logrmm = self.logrmax - self.logrmin

    def transform(self, xlg):
        return torch.exp((xlg + 1)*0.5 * self.logrmm + self.logrmin) - self.rmin

    def invtransform(self, rs):
        return (torch.log(rs + self.rmin) - self.logrmin) / (0.5 * self.logrmm) - 1.0

    def get_scaling(self, rs):
        return (rs + self.rmin) * self.logrmm * 0.5

    #################### editable module parts ####################
    def getparamnames(self, methodname, prefix=""):
        if methodname == "invtransform" or methodname == "transform":
            return [prefix+"logrmin", prefix+"logrmm"]
        elif methodname == "get_scaling":
            return [prefix+"logrmm"]
        else:
            return super().getparamnames(methodname, prefix=prefix)

class DoubleExp2Transformation(BaseGridTransformation):
    def __init__(self, alpha, rmin, rmax, dtype=torch.float, device=torch.device('cpu')):
        # setup the parameters needed for the transformation
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=dtype, device=device)
        self.alpha = alpha
        self.xmin = self.rtox(torch.tensor([rmin], dtype=dtype, device=device))
        self.xmax = self.rtox(torch.tensor([rmax], dtype=dtype, device=device))

    def transform(self, xlg):
        x = (xlg+1)*0.5 * (self.xmax - self.xmin) + self.xmin
        rs = torch.exp(self.alpha*x - torch.exp(-x))
        return rs

    def invtransform(self, rs):
        x = self.rtox(rs)
        xlg = (x - self.xmin) / (self.xmax - self.xmin) * 2 - 1
        return xlg

    def rtox(self, rs):
        logrs = torch.log(rs)
        def iter_fcn(x, logrs, inv_alpha):
            return inv_alpha * (logrs + torch.exp(-x))
        x0 = torch.zeros_like(rs).to(rs.device)

        # equilibrium works with batching, so append the first dimension
        x = lintorch.optimize.equilibrium(iter_fcn, x0.unsqueeze(0),
            params=[logrs.unsqueeze(0), 1./self.alpha.unsqueeze(0)],
            fwd_options={"method": "np_broyden1"}).squeeze(0)
        return x

    def get_scaling(self, rs):
        x = self.rtox(rs)
        return rs * (self.alpha + torch.exp(-x)) * 0.5 * (self.xmax - self.xmin)

    #################### editable module parts ####################
    def getparamnames(self, methodname, prefix=""):
        if methodname == "invtransform" or methodname == "transform" or methodname == "get_scaling":
            return [prefix+"alpha"]
        else:
            return super().getparamnames(methodname, prefix=prefix)

    # def getparams(self, methodname):
    #     if methodname == "invtransform" or methodname == "transform" or methodname == "get_scaling":
    #         return [self.alpha]
    #     else:
    #         raise RuntimeError("Unimplemented %s method for getparams" % methodname)
    #
    # def setparams(self, methodname, *params):
    #     if methodname == "invtransform" or methodname == "transform" or methodname == "get_scaling":
    #         self.alpha, = params[:1]
    #         return 1
    #     else:
    #         raise RuntimeError("Unimplemented %s method for setparams" % methodname)

if __name__ == "__main__":
    import lintorch as lt
    grid = LegendreShiftExpRadGrid(100, 1e-4, 1e2, dtype=torch.float64)
    rgrid = grid.rgrid.clone().detach()
    f = torch.exp(-rgrid[:,0].unsqueeze(0)**2*0.5)
    print(f)

    lt.list_operating_params(grid.solve_poisson, f)
    lt.list_operating_params(grid.interpolate, f, rgrid)
    lt.list_operating_params(grid.get_dvolume)
