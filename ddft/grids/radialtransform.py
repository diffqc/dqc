import torch
import numpy as np
import lintorch as lt
from ddft.grids.base_grid import BaseTransformed1DGrid

class ShiftExp(BaseTransformed1DGrid):
    def __init__(self, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
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
    def __init__(self, alpha, rmin, rmax, nr, dtype=torch.float, device=torch.device('cpu')):
        # setup the parameters needed for the transformation
        self.alpha = alpha
        self.xmin = self.invtransform(rmin)
        self.xmax = self.invtransform(rmax)
        self.x = torch.linspace(self.xmin, self.xmax, nr, dtype=dtype, device=device)

    def transform(self, xlg):
        x = (xlg+1)*0.5 * (self.xmax - self.xmin) + self.xmin
        return torch.exp(self.alpha*x - torch.exp(-x))

    def invtransform(self, rs):
        logrs = torch.log(rs)
        def iter_fcn(x, logrs, inv_alpha):
            return inv_alpha * (logrs + torch.exp(-x))
        x0 = torch.zeros_like(rs).to(rs.device)
        x = lt.equilibrium(iter_fcn, x0, params=[logrs, 1./self.alpha])
        return x

    def get_scaling(self, rs):
        x = self.invtransform(rs)
        return rs * (self.alpha - torch.exp(-x))

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
