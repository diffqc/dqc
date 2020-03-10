from abc import abstractmethod
import torch
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from ddft.grids.base_grid import BaseGrid, BaseRadialGrid
from ddft.utils.legendre import legint

class LegendreRadialTransform(BaseRadialGrid):
    def __init__(self, nx, dtype=torch.float, device=torch.device('cpu')):
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
        basis = legvander(xleggauss, nx-1) # (nr, nr)
        self.basis = torch.tensor(basis.T).to(dtype).to(device)
        self.inv_basis = self.basis.inverse()

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

    def antiderivative(self, intgn, dim=-1, zeroat="left"):
        # intgn: (..., nr, ...)
        intgn = intgn.transpose(dim, -1) # (..., nr)
        intgn = intgn * self._scaling
        coeff = torch.matmul(intgn, self.inv_basis) # (..., nr)
        intcoeff = legint(coeff, dim=dim, zeroat=zeroat)[..., :-1] # (nbatch, nr)
        res = torch.matmul(intcoeff, self.basis) # (..., nr)
        res = res.transpose(dim, -1)
        return res

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

    def get_scaling(self, rs):
        return (rs + self.rmin) * self.logrmm * 0.5
