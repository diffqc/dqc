from abc import abstractmethod
import torch
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from ddft.grids.base_grid import BaseGrid, BaseRadialGrid
from ddft.utils.legendre import legint

class RadialShiftExp(BaseRadialGrid):
    def __init__(self, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
        logr = torch.linspace(np.log(rmin), np.log(rmax), nr).to(dtype).to(device)
        unshifted_rgrid = torch.exp(logr)
        eps = 1e-12
        self._boxshape = (nr,)
        self.rmin = unshifted_rgrid[0]
        self.rs = unshifted_rgrid - self.rmin + eps # eps for safeguarding not to touch negative
        self._rgrid = self.rs.unsqueeze(1) # (nr, 1)
        self.dlogr = logr[1] - logr[0]

        # integration elements
        self._scaling = (self.rs + self.rmin)
        self._dr = self._scaling * self.dlogr
        self._dvolume = 4 * np.pi * self.rs*self.rs * self._dr

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # the expression below is used to make the operator symmetric
        eps = 1e-10
        intgn1 = f * self.rs * self.rs
        int1 = self.antiderivative(intgn1, dim=-1, zeroat="left")
        intgn2 = int1 / (self.rs * self.rs + eps)
        # this form of cumsum is the transpose of torch.cumsum
        int2 = self.antiderivative(intgn2, dim=-1, zeroat="right")
        return -int2

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape

    def antiderivative(self, intgn, dim=-1, zeroat="left"):
        intgn = intgn * self._dr
        if zeroat == "left":
            return torch.cumsum(intgn, dim=dim)
        elif zeroat == "right":
            return torch.cumsum(intgn.flip(dims=[dim]), dim=dim).flip(dims=[dim])

class LegendreRadialShiftExp(BaseRadialGrid):
    def __init__(self, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
        self._boxshape = (nr,)
        self.rmin = rmin

        # setup the legendre points
        logrmin = torch.tensor(np.log(rmin)).to(dtype).to(device)
        logrmax = torch.tensor(np.log(rmax)).to(dtype).to(device)
        self.logrmm = logrmax - logrmin
        xleggauss, wleggauss = leggauss(nr)
        self.xleggauss = torch.tensor(xleggauss, dtype=dtype, device=device)
        self.wleggauss = torch.tensor(wleggauss, dtype=dtype, device=device)
        self.rs = torch.exp((self.xleggauss+1)*0.5 * (logrmax - logrmin) + logrmin) - rmin
        self._rgrid = self.rs.unsqueeze(-1)

        # integration elements
        self._scaling = (self.rs+self.rmin) * self.logrmm * 0.5 # dr/dg
        self._dr = self._scaling * self.wleggauss
        self._dvolume = (4*np.pi*self.rs*self.rs) * self._dr

        # legendre basis (from tinydft/tinygrid.py)
        basis = legvander(xleggauss, nr-1) # (nr, nr)
        # U, S, Vt = np.linalg.svd(basis)
        # inv_basis = np.einsum('ji,j,kj->ik', Vt, 1 / S, U) # (nr, nr)
        self.basis = torch.tensor(basis.T).to(dtype).to(device)
        self.inv_basis = self.basis.inverse()
        # self.inv_basis = torch.tensor(inv_basis.T).to(dtype).to(device)

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # the expression below is used to make the operator symmetric

        # calculate the matrix rless / rgreat
        rless = torch.min(self.rs.unsqueeze(-1), self.rs) # (nr, nr)
        rgreat = torch.max(self.rs.unsqueeze(-1), self.rs)
        rratio = (1. / rgreat) # (nr, nr)

        # the integralbox for radial grid is integral[4*pi*r^2 f(r) dr] while here
        # we only need to do integral[f(r) dr]. That's why it is divided by (4*np.pi)
        # and it is not multiplied with (self.radrgrid**2) in the lines below
        intgn = (f).unsqueeze(-2) * rratio # (nbatch, nr, nr)
        vrad_lm = self.integralbox(intgn / (4*np.pi), dim=-1)
        return -vrad_lm

        # eps = 1e-12
        # intgn1 = f * self.rs * self.rs
        # int1 = self.antiderivative(intgn1, dim=-1, zeroat="left")
        # intgn2 = int1 / (self.rs * self.rs + eps)
        # # this form of cumsum is the transpose of torch.cumsum
        # int2 = self.antiderivative(intgn2, dim=-1, zeroat="right")
        # return -int2

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape

    def antiderivative(self, intgn, dim=-1, zeroat="left"):
        # intgn: (..., nr, ...)
        intgn = intgn.transpose(dim, -1) # (..., nr)
        intgn = intgn * self._scaling
        coeff = torch.matmul(intgn, self.inv_basis) # (..., nr)
        intcoeff = legint(coeff, dim=dim, zeroat=zeroat)[..., :-1] # (nbatch, nr)
        res = torch.matmul(intcoeff, self.basis) # (..., nr)
        res = res.transpose(dim, -1)
        return res
