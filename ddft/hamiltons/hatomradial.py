from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import numpy as np
import lintorch as lt

from ddft.hamiltons.base_hamilton import BaseHamilton

class HamiltonAtomRadial(BaseHamilton):
    """
    HamiltonAtomRadial represents the system of one atom with all-electrons
    potential at the centre of coordinate. The chosen basis is
    well-tempered Gaussian radial basis set.
    This Hamiltonian only works for symmetric density and potential, so it only
    works for s-only atoms and closed-shell atoms
    (e.g. H, He, Li, Be, Ne, Ar)

    Arguments
    ---------
    * grid: BaseGrid
        The integration grid.
    * gwidths: torch.tensor (ng,)
        The tensor of Gaussian-widths of the basis. The tensor should be uniform
        in logspace.
    * angmom: int
        The angular momentum of the Hamiltonian

    Forward arguments
    -----------------
    * wf: torch.tensor (nbatch, ns, ncols)
        The basis coefficients of the radial wavefunction at the given
        angular momentum.
    * vext: torch.tensor (nbatch, nr)
        External radial potential other than the potential from the central atom.
    * atomz: torch.tensor (nbatch,)
        The atomic number of the central atom.
    * coulexp: bool
        If True, the coulomb potential is Z/r*exp(-r*r). If False, it is Z/r.
        Default True.
        If True, then the remaining coulomb potential part is added as external
        potential internally in this object during the forward evaluation.
        The reason of using this option is to avoid the long tail of Hartree
        potential.

    Overlap arguments
    -----------------
    * wf: torch.tensor (nbatch, ns, ncols)
        The basis coefficients of the radial wavefunction at the given
        angular momentum.

    Note
    ----
    * To get the best accuracy, the gaussian width range should be well inside
        the radial grid range.
    """

    def __init__(self, grid, gwidths,
                       angmom=0,coulexp=True):
        ng = gwidths.shape[0]
        self._grid = grid
        super(HamiltonAtomRadial, self).__init__(
            shape = (ng, ng),
            is_symmetric = True,
            is_real = True)

        # well-tempered gaussian factor from tinydft
        self.gwidths = gwidths # torch.nn.Parameter(gwidths) # (ng)
        self.rs = grid.rgrid[:,0] # (nr,)
        self.angmom = angmom
        self.use_coulexp = coulexp

        # get the basis in rgrid
        # (ng, nr)
        gw1 = self.gwidths.unsqueeze(-1) # (ng, 1)
        unnorm_basis = torch.exp(-self.rs*self.rs / (2*gw1*gw1)) * self.rs # (nr,)
        norm = np.sqrt(2./3) / gw1**2.5 / np.pi**.75 # (ng, 1)
        self.basis = norm * unnorm_basis # (ng, nr)

        # print(self.grid.integralbox(self.basis*self.basis))
        # raise RuntimeError

        # construct the matrices provided ng is small enough
        gwprod = gw1 * self.gwidths
        gwprod32 = gwprod**1.5
        gwprod2 = gwprod*gwprod
        gwprod12 = gwprod**0.5
        gwprod52 = gwprod**2.5
        gw2sum = gw1*gw1 + self.gwidths*self.gwidths
        gwnet2 = gwprod*gwprod / gw2sum
        gwnet = torch.sqrt(gwnet2)
        gwpoly = 2*(gw1**4 + self.gwidths**4) - 11*(gw1*gw1)*(self.gwidths*self.gwidths)

        # kin_rad = 3.0/np.sqrt(2.0) * gwnet**3 / gwprod32 / gw2sum
        # kin_ang = gwnet / gwprod32 / np.sqrt(2.0)
        # kin = kin_rad + kin_ang * angmom * (angmom+1)
        # olp = 2*np.sqrt(2.0) * gwnet**3 / gwprod32
        # coul = -4.0 / np.sqrt(2.0 * np.pi) * gwnet2 / gwprod32

        olp = 4 * np.sqrt(2) * gwnet**5 / gwprod52
        coul = -16./(3*np.sqrt(np.pi)) * gwnet**4 / gwprod52
        kin_ang = 2 * np.sqrt(2) / 3 * gwnet**3 / gwprod52
        kin_rad = -2*np.sqrt(2) / 3 * gwnet**3 / gw2sum**2 / gwprod52 * gwpoly
        kin = kin_rad + kin_ang * angmom * (angmom+1)

        if self.use_coulexp:
            coulexp = -16.0 / (3*np.sqrt(np.pi) * gwprod52 * (2+1./gwnet2)**2)
            self.coulexp = coulexp.unsqueeze(0)

            # the remaining part of the coulomb potential
            self.vext_coulexp = -1./self.rs * (1 - torch.exp(-self.rs*self.rs))
            smidx = self.rs < 1e-3
            smrs = self.rs[smidx]
            self.vext_coulexp[smidx] = -smrs * (1.0 - smrs*smrs*0.5)
            self.vext_coulexp = self.vext_coulexp.unsqueeze(0) # (1,nr)

        # create the batch dimension to the matrix to enable batched matmul
        # shape: (1, ns, ns)
        self.kin = kin.unsqueeze(0)
        self.olp = olp.unsqueeze(0)
        self.coul = coul.unsqueeze(0)

    ############################# basis part #############################
    @property
    def nhparams(self):
        return 2 # atomz & charge

    def forward(self, wf, vext, atomz, charge):
        # wf: (nbatch, ns, ncols)
        # vext: (nbatch, nr)
        # atomz: (nbatch,)
        # charge: (nbatch,)

        fock = self.fullmatrix(vext, atomz, charge)
        hwf = torch.bmm(fock, wf)
        return hwf

    def precond(self, y, vext, atomz, biases=None, M=None, mparams=None):
        return y # ???

    def fullmatrix(self, vext, atomz, charge):
        # the external potential part
        numel = atomz - charge
        if self.use_coulexp:
            vext = vext + self.vext_coulexp * numel.unsqueeze(-1)
        extpot = self.grid.mmintegralbox(vext.unsqueeze(1) * self.basis, self.basis.transpose(-2,-1))

        # the coulomb part
        if self.use_coulexp:
            pure_coul = self.coul * charge.unsqueeze(-1).unsqueeze(-1)
            coulexp = self.coulexp * numel.unsqueeze(-1).unsqueeze(-1)
            coul = pure_coul + coulexp
        else:
            coul = self.coul * atomz.unsqueeze(-1).unsqueeze(-1)

        # add all the matrix and apply the Hamiltonian
        fock = self.kin + extpot + coul
        return fock

    def _overlap(self, wf):
        return torch.matmul(self.olp, wf)

    def torgrid(self, wfs, dim=-2):
        # wfs: (..., ng, ...)
        wfs = wfs.transpose(dim, -1) # (..., ns)
        wfr = torch.matmul(wfs, self.basis) # (..., nr)
        return wfr.transpose(dim, -1)

    ############################# grid part #############################
    @property
    def grid(self):
        return self._grid

    ########################## editable module part #######################
    def getparams(self, methodname):
        methods = ["fullmatrix", "forward", "__call__", "transpose"]
        if methodname in methods:
            res = [self.coulexp, self.vext_coulexp] if self.use_coulexp else []
            res = res + [self.basis, self.coul, self.kin] + self.grid.getparams("get_dvolume")
            return res
        elif methodname == "_overlap":
            return [self.olp]
        elif methodname == "torgrid":
            return [self.basis]
        else:
            return super().getparams(methodname)

    def setparams(self, methodname, *params):
        methods = ["fullmatrix", "forward", "__call__", "transpose"]
        if methodname in methods:
            idx = 0
            if self.use_coulexp:
                self.coulexp, self.vext_coulexp = params[:2]
                idx += 2
            self.basis, self.coul, self.kin = params[idx:idx+3]
            idx += 3
            idx += self.grid.setparams("get_dvolume", *params[idx:])
            return idx
        elif methodname == "_overlap":
            self.olp, = params[:1]
            return 1
        elif methodname == "torgrid":
            self.basis, = params[:1]
            return 1
        else:
            return super().setparams(methodname, *params)

if __name__ == "__main__":
    from ddft.grids.radialgrid import LegendreShiftExpRadGrid
    dtype = torch.float64
    gwidths = torch.logspace(np.log10(1e-5), np.log10(1e2), 100).to(dtype)
    grid = LegendreShiftExpRadGrid(200, 1e-6, 1e4, dtype=dtype)
    h = HamiltonAtomRadial(grid, gwidths, angmom=0).to(dtype)

    vext = torch.zeros(1, 2000).to(dtype)
    atomz = torch.tensor([1.0]).to(dtype)
    H = h.fullmatrix(vext, atomz)
    olp = h.olp
    print(torch.symeig(olp)[0])
    evals, evecs = torch.eig(torch.solve(H[0], olp[0])[0])
    print(evals)
