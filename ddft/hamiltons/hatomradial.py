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
    well-tempered Gaussian radial basis set and the chosen grid is shifted
    exponential in radial direction.
    This Hamiltonian only works for symmetric density and potential, so it only
    works for s-only atoms and closed-shell atoms
    (e.g. H, He, Li, Be, Ne, Ar)

    Arguments
    ---------
    * gwidths: torch.tensor (ng,)
        The tensor of Gaussian-widths of the basis. The tensor should be uniform
        in logspace.
    * rs: torch.tensor (nr,)
        The tensor of the biased radial grid. The tensor should be uniform
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

    def __init__(self, gwidths, rs,
                       angmom=0):
        ng = gwidths.shape[0]
        super(HamiltonAtomRadial, self).__init__(
            shape = (ng, ng),
            is_symmetric = True,
            is_real = True)

        # well-tempered gaussian factor from tinydft
        self.gwidths = gwidths # (ng)
        self.gs = 0.5 / (self.gwidths * self.gwidths)
        self.logrs = torch.log(rs) # (nr)
        self.rmin = rs[0]
        self.dlogr = self.logrs[1] - self.logrs[0]
        self.rs = rs - self.rmin
        self.angmom = angmom

        # get the basis in rgrid
        # (ng, nr)
        gw1 = self.gwidths.unsqueeze(-1)
        unnorm_basis = torch.exp(-self.rs*self.rs / (4*gw1*gw1))
        norm = 1. / (np.sqrt(2 * np.pi) * self.gwidths.unsqueeze(-1))**(1.5) # (ng, 1)
        self.basis = norm * unnorm_basis

        # print(self.integralbox(self.basis*self.basis))

        # construct the matrices provided ng is small enough
        gwprod = gw1 * self.gwidths
        gwprod32 = gwprod**1.5
        gw2sum = gw1*gw1 + self.gwidths*self.gwidths
        gwnet2 = gwprod*gwprod / gw2sum
        gwnet = torch.sqrt(gwnet2)

        kin_rad = 3.0/np.sqrt(2.0) * gwnet**3 / gwprod32 / gw2sum
        kin_ang = gwnet / gwprod32 / np.sqrt(2.0)
        kin = kin_rad + kin_ang * angmom * (angmom+1)
        olp = 2*np.sqrt(2.0) * gwnet**3 / gwprod32
        coul = -4.0 / np.sqrt(2.0 * np.pi) * gwnet2 / gwprod32

        # create the batch dimension to the matrix to enable batched matmul
        # shape: (1, ns, ns)
        self.kin = kin.unsqueeze(0)
        self.olp = olp.unsqueeze(0)
        self.coul = coul.unsqueeze(0)

    ############################# basis part #############################
    def forward(self, wf, vext, atomz):
        # wf: (nbatch, ns, ncols)
        # vext: (nbatch, nr)
        # atomz: (nbatch,)

        # the external potential part
        zeros = torch.zeros(1, dtype=vext.dtype).to(vext.device)
        if torch.allclose(vext, zeros):
            extpot = 0
        else:
            raise RuntimeError("Nonzero external potential has not been implemented.")

        # add all the matrix and apply the Hamiltonian
        fock = (self.kin + extpot) + self.coul * atomz.unsqueeze(-1).unsqueeze(-1)
        hwf = torch.matmul(fock, wf)
        return hwf

    def precond(self, y, vext, atomz, biases=None, M=None, mparams=None):
        return y # ???

    def overlap(self, wf):
        return torch.matmul(self.olp, wf)

    def tocoeff(self, wfr, dim=-2):
        pass # ???

    def torgrid(self, wfs, dim=-2):
        # wfs: (..., ng, ...)
        ndim = wfs.ndim
        if dim < 0:
            dim = ndim + dim

        # unsqueeze basis ndim-dim-1 times
        nunsq = ndim - dim - 1
        basis = self.basis
        for _ in range(nunsq):
            basis = basis.unsqueeze(-1) # (nr, ...)
        wfr = (wfs.unsqueeze(dim+1) * basis).sum(dim=dim) # (..., nr, ...)
        return wfr

    ############################# grid part #############################
    def rgrid(self):
        return self.rs

    def getvhartree(self, dens):
        raise RuntimeError("getvhartree for HamiltonAtomRadial has not been implemented.")

    def integralbox(self, p, dim=-1):
        # p: (nbatch, nr)
        # return (nbatch,)
        integrand = p * (self.rs + self.rmin) * 4 * np.pi * self.rs*self.rs
        res = torch.sum(integrand, dim=dim) * self.dlogr
        return res

if __name__ == "__main__":
    dtype = torch.float64
    gwidths = torch.logspace(np.log10(1e-5), np.log10(1e2), 100).to(dtype)
    rs = torch.logspace(np.log10(1e-6), np.log10(1e4), 2000).to(dtype)
    h = HamiltonAtomRadial(gwidths, rs, angmom=0)

    vext = torch.zeros(1, 2000).to(dtype)
    atomz = torch.tensor([1.0]).to(dtype)
    H = h.fullmatrix(vext, atomz)
    olp = h.olp
    print(torch.symeig(olp)[0])
    evals, evecs = torch.eig(torch.solve(H[0], olp[0])[0])
    print(evals)
