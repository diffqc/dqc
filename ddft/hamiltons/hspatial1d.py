import torch
from ddft.hamiltons.base_hamilton import BaseHamilton

class HamiltonSpatial1D(BaseHamilton):
    def __init__(self, rgrid):
        super(HamiltonSpatial1D, self).__init__()
        self._rgrid = rgrid # (nr)
        self.dr = rgrid[1] - rgrid[0]
        self.inv_dr = 1.0 / self.dr
        self.inv_dr2 = self.inv_dr * self.inv_dr
        nr = len(rgrid)
        self.Kdiag = torch.ones(nr).to(rgrid.dtype).to(rgrid.device)
        self._boxshape = (nr,)
        self._shape = (nr,nr)

    def kinetics(self, wf):
        return (wf - (torch.roll(wf,1,dims=1) + torch.roll(wf,-1,dims=1)) * 0.5) * self.inv_dr2 # (nbatch, nr, ncols)

    def kinetics_diag(self, nbatch):
        return self.Kdiag.unsqueeze(0).expand(nbatch,-1)

    def getdens(self, eigvec2):
        return eigvec2 * self.inv_dr

    def integralbox(self, p, dim=-1):
        return p.sum(dim=dim) * self.dr

    @property
    def shape(self):
        return self._shape
