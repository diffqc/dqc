import torch
import numpy as np
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
        if methodname == "invtransform":
            return [self.logrmin, self.logrmm]
        else:
            raise RuntimeError("Unimplemented %s method for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "invtransform":
            self.logrmin, self.logrmm = params[:2]
            return 2
        else:
            raise RuntimeError("Unimplemented %s method for setparams" % methodname)
