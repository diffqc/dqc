from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import numpy as np
import lintorch as lt

from ddft.hamiltons.base_hamilton import BaseHamilton
from ddft.utils.spharmonics import spharmonics

class HamiltonMoleculeC0Gauss(BaseHamilton):
    """
    HamiltonMoleculeC0Gauss represents the system of multiple atoms with
    all-electrons potential at the centre of coordinate. The chosen basis is
    contracted Cartesian Gaussian with total ijk == 0 centered at given positions.
    As the basis depends on the position of each atom, the batch only works
    for molecules with the same positions.

    Arguments
    ---------
    * grid: BaseGrid
        The integration grid. It should be a spherical harmonics radial grid
        centered at the centre of coordinate.
    * alphas: torch.tensor (nbasis, nelmts,)
        The Gaussian exponent of the contracted Gaussians.
    * centres: torch.tensor (nbasis, nelmts, 3)
        The position in Cartesian coordinate of the contracted Gaussians.
    * coeffs: torch.tensor (nbasis, nelmts,)
        The contracted coefficients of each elements in the contracted Gaussians.
    * atompos: torch.tensor (natoms, 3)
        The position of the atoms (to be the central position of Coulomb potential).
    * atomzs: torch.tensor (natoms,)
        The atomic number of the atoms.
    * coulexp: bool
        If True, the coulomb potential is Z/r*exp(-r*r). If False, it is Z/r.
        Default True.
        If True, then the remaining coulomb potential part is added as external
        potential internally in this object during the forward evaluation.
        The reason of using this option is to avoid the long tail of Hartree
        potential.

    Forward arguments
    -----------------
    * wf: torch.tensor (nbatch, nbasis, ncols)
        The basis coefficients of the wavefunction.
    * vext: torch.tensor (nbatch, nr)
        External potential other than the potential from the atoms.

    Overlap arguments
    -----------------
    * wf: torch.tensor (nbatch, nbasis, ncols)
        The basis coefficients of the wavefunction.
    """

    def __init__(self, grid,
                 alphas, centres, coeffs,
                 atompos, atomzs,
                 coulexp=True):
        self.nbasis, self.nelmts, ndim = centres.shape
        assert ndim == 3, "The centres must be 3 dimensions"
        self.natoms = atompos.shape[0]
        self._grid = grid
        dtype = alphas.dtype
        device = alphas.device
        super(HamiltonMoleculeC0Gauss, self).__init__(
            shape = (self.nbasis, self.nbasis),
            is_symmetric = True,
            is_real = True,
            dtype = dtype,
            device = device)

        # flatten the information about the basis
        alphas = alphas.view(-1) # (nbasis*nelmts,)
        centres = centres.view(-1, ndim) # (nbasis*nelmts, 3)
        coeffs = coeffs.view(-1) # (nbasis*nelmts,)
        self.coeffs = coeffs

        # prepare the basis
        # get the frequently used variables
        qab_sq = ((centres - centres.unsqueeze(1))**2).sum(dim=-1) # (nbasis*nelmts, nbasis*nelmts)
        gamma = alphas + alphas.unsqueeze(1) # (nbasis*nelmts, nbasis*nelmts)
        kappa = alphas * alphas.unsqueeze(1) / gamma # (nbasis*nelmts, nbasis*nelmts)
        # print(alphas.shape, gamma.shape, kappa.shape, qab_sq.shape)
        mab = torch.exp(-kappa * qab_sq) # (nbasis*nelmts, nbasis*nelmts)
        ra = alphas.unsqueeze(-1) * centres # (nbasis*nelmts, 3)
        rc = (ra + ra.unsqueeze(1)) / gamma.unsqueeze(-1) # (nbasis*nelmts, nbasis*nelmts, 3)
        self.coeffs2 = coeffs * coeffs.unsqueeze(1) # (nbasis*nelmts, nbasis*nelmts)

        # overlap part
        olp = (mab * (np.pi/gamma)**1.5) # (nbasis*nelmts, nbasis*nelmts)
        self.olp_elmts = olp.unsqueeze(0) # (1, nbasis*nelmts, nbasis*nelmts)

        # kinetics part
        kin = olp * kappa * (3 - 2*kappa*qab_sq) # (nbasis*nelmts, nbasis*nelmts)
        self.kin_elmts = kin.unsqueeze(0) # (1, nbasis*nelmts, nbasis*nelmts)

        # coulomb part
        rcd = (rc - atompos.unsqueeze(-2).unsqueeze(-2) + 1e-12) # (natoms, nbasis*nelmts, nbasis*nelmts, 3)
        q0 = torch.sqrt((rcd*rcd).sum(dim=-1)) # (natoms, nbasis*nelmts, nbasis*nelmts)
        coul = -olp * (torch.erf(torch.sqrt(gamma) * (q0+1e-12)) / (q0+1e-12)) # (natoms, nbasis*nelmts, nbasis*nelmts)
        coul_small = olp * torch.sqrt(gamma) * 2/np.sqrt(np.pi)
        self.coul_elmts = coul * atomzs.unsqueeze(-1).unsqueeze(-1) # (natoms, nbasis*nelmts, nbasis*nelmts)

        # combine the kinetics and coulomb elements
        self.kin_coul_elmts = self.kin_elmts + self.coul_elmts.sum(dim=0, keepdim=True) # (1, nbasis*nelmts, nbasis*nelmts)

        # get the contracted part
        self.kin_coul_mat = self._contract(self.kin_coul_elmts) # (1, nbasis, nbasis)
        self.olp_mat = self._contract(self.olp_elmts) # (1, nbasis, nbasis)

        # get the normalization constant
        norm = 1. / torch.sqrt(self.olp_mat.diagonal(dim1=-2, dim2=-1)) # (1,nbasis)
        norm_mat = norm.unsqueeze(-1) * norm.unsqueeze(1) # (1, nbasis, nbasis)

        # normalize the contracted matrix
        self.kin_coul_mat = self.kin_coul_mat * norm_mat
        self.olp_mat = self.olp_mat * norm_mat

        # get the basis
        self.rgrid = self.grid.rgrid_in_xyz # (nr, 3)
        dist_sq = ((self.rgrid - centres.unsqueeze(1))**2).sum(dim=-1) # (nbasis*nelmts, nr)
        basis_all = torch.exp(-alphas.unsqueeze(-1) * dist_sq) # (nbasis*nelmts, nr)
        basis = (basis_all * coeffs.unsqueeze(-1)).view(self.nbasis, self.nelmts, -1).sum(dim=1) # (nbasis, nr)
        norm_basis = basis * norm.squeeze(0).unsqueeze(-1)
        self.basis = norm_basis # (nbasis, nr)
        self.basis_dvolume = self.basis * self.grid.get_dvolume() # (nbasis, nr)

    ############################# basis part #############################
    @property
    def nhparams(self):
        return 0

    def forward(self, wf, vext):
        # wf: (nbatch, nbasis, ncols)
        # vext: (nbatch, nr)

        nbatch, ns, ncols = wf.shape
        kin_coul = torch.matmul(self.kin_coul_mat, wf) # (nbatch, nbasis, ncols)

        # vext part
        # self.basis: (nbasis, nr)
        # extpot: (nbatch, nbasis, nbasis)
        extpot = torch.matmul(vext.unsqueeze(1) * self.basis_dvolume, self.basis.transpose(-2,-1))
        extpot = torch.bmm(extpot, wf) # (nbatch, nbasis, ncols)

        hwf = extpot + kin_coul
        return hwf

    def precond(self, y, vext, atomz, biases=None, M=None, mparams=None):
        # y: (nbatch, nbasis, ncols)
        # biases: (nbatch, ncols)
        return y

    def _overlap(self, wf):
        nbatch, nbasis, ncols = wf.shape
        res = torch.matmul(self.olp_mat, wf)  # (nbatch, nbasis, ncols)
        return res

    def torgrid(self, wfs, dim=-2):
        # wfs: (..., nbasis, ...)
        wfs = wfs.transpose(dim, -1) # (..., nbasis)
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
            return [self.kin_coul_mat, self.basis_dvolume, self.basis]
        elif methodname == "_overlap":
            return [self.olp_mat]
        elif methodname == "torgrid":
            return [self.basis]
        else:
            return super().getparams(methodname)

    def setparams(self, methodname, *params):
        methods = ["fullmatrix", "forward", "__call__", "transpose"]
        if methodname in methods:
            self.kin_coul_mat, self.basis_dvolume, self.basis = params[:3]
            return 3
        elif methodname == "_overlap":
            self.olp_mat, = params[:1]
            return 1
        elif methodname == "torgrid":
            self.basis, = params[:1]
            return 1
        else:
            return super().setparams(methodname, *params)

    ############################# helper functions #############################
    def _contract(self, mat):
        # multiply the matrix with the contracted coefficients
        mat = mat * self.coeffs2

        # resize mat to have shape of (-1, nbasis, nelmts, nbasis, nelmts)
        batch_size = mat.shape[:-2]
        mat_size = mat.shape[-2:]
        mat = mat.view(-1, self.nbasis, self.nelmts, self.nbasis, self.nelmts)

        # sum the nelmts to get the basis
        cmat = mat.sum(dim=-1).sum(dim=-2) # (-1, nbasis, nbasis)
        return cmat.view(*batch_size, self.nbasis, self.nbasis)

if __name__ == "__main__":
    from ddft.grids.radialgrid import LegendreRadialShiftExp
    from ddft.grids.sphangulargrid import Lebedev
    from ddft.grids.multiatomsgrid import BeckeMultiGrid

    dtype = torch.float64
    atompos = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype) # (natoms, ndim)
    atomzs = torch.tensor([1.0], dtype=dtype)
    radgrid = LegendreRadialShiftExp(1e-6, 1e3, 200, dtype=dtype)
    atomgrid = Lebedev(radgrid, prec=13, basis_maxangmom=4, dtype=dtype)
    grid = BeckeMultiGrid(atomgrid, atompos, dtype=dtype)
    nr = grid.rgrid.shape[0]

    nbasis = 30
    alphas = torch.logspace(np.log10(1e-4), np.log10(1e6), nbasis).unsqueeze(-1).to(dtype) # (nbasis, 1)
    centres = atompos.unsqueeze(1).repeat(nbasis, 1, 1)
    coeffs = torch.ones((nbasis, 1))
    h = HamiltonMoleculeC0Gauss(grid, alphas, centres, coeffs, atompos, atomzs, False).to(dtype)

    vext = torch.zeros(1, nr).to(dtype)
    H = h.fullmatrix(vext)
    olp = h.overlap.fullmatrix()
    # check symmetricity
    assert torch.allclose(olp-olp.transpose(-2,-1), torch.zeros_like(olp))
    assert torch.allclose(H-H.transpose(-2,-1), torch.zeros_like(H))
    print(torch.symeig(olp)[0])
    print(torch.symeig(H)[0])
    mat = torch.solve(H[0], olp[0])[0]
    evals, evecs = torch.eig(mat)
    evals = torch.sort(evals.view(-1))[0]
    print(evals[:20])
