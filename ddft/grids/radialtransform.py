import torch
import numpy as np
import lintorch as lt
from ddft.grids.base_grid import BaseTransformed1DGrid

class ShiftExp(BaseTransformed1DGrid):
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
    def getparams(self, methodname):
        if methodname == "invtransform" or methodname == "transform":
            return [self.logrmin, self.logrmm]
        elif methodname == "get_scaling":
            return [self.logrmm]
        else:
            raise RuntimeError("Unimplemented %s method for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "invtransform" or methodname == "transform":
            self.logrmin, self.logrmm = params[:2]
            return 2
        elif methodname == "get_scaling":
            self.logrmm, = params[:1]
            return 1
        else:
            raise RuntimeError("Unimplemented %s method for setparams" % methodname)

class DoubleExp2(BaseTransformed1DGrid):
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

        # lt.equilibrium works with batching, so append the first dimension
        x = lt.equilibrium(iter_fcn, x0.unsqueeze(0),
            params=[logrs.unsqueeze(0), 1./self.alpha.unsqueeze(0)],
            fwd_options={"method": "np_broyden1"}).squeeze(0)
        return x

    def get_scaling(self, rs):
        x = self.rtox(rs)
        return rs * (self.alpha + torch.exp(-x)) * 0.5 * (self.xmax - self.xmin)

    #################### editable module parts ####################
    def getparams(self, methodname):
        if methodname == "invtransform" or methodname == "transform" or methodname == "get_scaling":
            return [self.alpha]
        else:
            raise RuntimeError("Unimplemented %s method for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "invtransform" or methodname == "transform" or methodname == "get_scaling":
            self.alpha, = params[:1]
            return 1
        else:
            raise RuntimeError("Unimplemented %s method for setparams" % methodname)

class LogM3(BaseTransformed1DGrid):
    # eq (12) in https://aip.scitation.org/doi/pdf/10.1063/1.475719
    def __init__(self, ra=1.0, dtype=torch.float, device=torch.device('cpu')):
        # setup the parameters needed for the transformation
        if not isinstance(ra, torch.Tensor):
            ra = torch.tensor(ra, dtype=dtype, device=device)
        self.ra = ra
        self.ln2 = np.log(2.0)

    def transform(self, xlg):
        return self.ra * (1 - torch.log1p(-xlg) / self.ln2)

    def invtransform(self, rs):
        return -torch.expm1(self.ln2 * (1. - rs/self.ra))

    def get_scaling(self, rs):
        return self.ra / self.ln2 * torch.exp(-self.ln2 * (1. - rs / self.ra))

    #################### editable module parts ####################
    def getparams(self, methodname):
        if methodname == "invtransform" or methodname == "transform" or methodname == "get_scaling":
            return [self.ra]
        else:
            raise RuntimeError("Unimplemented %s method for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "invtransform" or methodname == "transform" or methodname == "get_scaling":
            self.ra, = params[:1]
            return 1
        else:
            raise RuntimeError("Unimplemented %s method for setparams" % methodname)
