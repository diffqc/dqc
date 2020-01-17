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

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape

    def kinetics(self, wf):
        return (wf - (torch.roll(wf,1,dims=1) + torch.roll(wf,-1,dims=1)) * 0.5) * self.inv_dr2 # (nbatch, nr, ncols)

    def kinetics_diag(self, nbatch):
        return self.Kdiag.unsqueeze(0).expand(nbatch,-1)

    def getdens(self, eigvec2):
        return eigvec2 * self.inv_dr

    def integralbox(self, p):
        return p.sum() * self.dr
