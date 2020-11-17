from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import numpy as np
import xitorch as xt

from ddft.hamiltons.base_hamilton_gen import BaseHamiltonGenerator, DensityInfo
from ddft.utils.spharmonics import spharmonics
from ddft.utils.gamma import incgamma
from ddft.csrc import get_ecoeff, get_overlap_mat, get_kinetics_mat, \
    get_coulomb_mat

class HamiltonMoleculeCGaussGenerator(BaseHamiltonGenerator):
    """
    HamiltonMoleculeCGaussGenerator represents the system of multiple atoms with
    all-electrons potential at the centre of coordinate. The chosen basis is
    contracted Cartesian Gaussian centered at given positions.
    As the basis depends on the position of each atom, the batch only works
    for molecules with the same positions.

    Arguments
    ---------
    * grid: BaseGrid
        The integration grid. It should be a spherical harmonics radial grid
        centered at the centre of coordinate.
    * ijks: torch.tensor int (nelmtstot, 3)
        The power of xyz in the basis.
    * alphas: torch.tensor (nelmtstot,)
        The Gaussian exponent of the contracted Gaussians.
    * centres: torch.tensor (nelmtstot, 3)
        The position in Cartesian coordinate of the contracted Gaussians.
    * coeffs: torch.tensor (nelmtstot,)
        The contracted coefficients of each elements in the contracted Gaussians.
    * nelmts: int or torch.tensor int (nbasis,)
        The number of elements per basis. If it is an int, then the number of
        elements are the same for all bases. If it is a tensor, then the number
        of elements in each basis is indicated by the values in the tensor.
    * atompos: torch.tensor (natoms, 3)
        The position of the atoms (to be the central position of Coulomb potential).
    * atomzs: torch.tensor (natoms,)
        The atomic number of the atoms.
    * normalize_elmts: bool
        If True, then the basis is `sum(coeff * norm(alpha) * xyz * exp(-alpha*r^2))`.
        Otherwise, then the basis is `sum(coeff * xyz * exp(-alpha*r^2))`.

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
                 ijks, alphas, centres, coeffs, nelmts,
                 atompos, atomzs, normalize_elmts=True):
        # self.nbasis, self.nelmts, ndim = centres.shape
        self.nelmtstot, ndim = centres.shape
        assert ndim == 3, "The centres must be 3 dimensions"
        self.nelmts = nelmts
        if isinstance(self.nelmts, torch.Tensor):
            self.nbasis = nelmts.shape[0]
        else:
            assert self.nelmtstot % self.nelmts == 0, "The number of gaussian is not the multiple of nelmts"
            self.nbasis = self.nelmtstot // self.nelmts
        self.natoms = atompos.shape[0]
        dtype = alphas.dtype
        device = alphas.device
        super(HamiltonMoleculeCGaussGenerator, self).__init__(
            grid = grid,
            shape = (self.nbasis, self.nbasis),
            dtype = dtype,
            device = device)

        # get the matrices before contraction
        ecoeff_obj = Ecoeff(ijks, centres, alphas, atompos)
        self.olp_elmts = ecoeff_obj.get_overlap() # (nelmtstot, nelmtstot)
        self.kin_elmts = ecoeff_obj.get_kinetics() # (nelmtstot, nelmtstot)
        self.coul_elmts = ecoeff_obj.get_coulomb() * atomzs.unsqueeze(-1).unsqueeze(-1) # (natoms, nelmtstot, nelmtstot)

        # normalize each gaussian elements
        if normalize_elmts:
            norm_elmt = 1./torch.sqrt(torch.diagonal(self.olp_elmts)) # (nelmtstot,)
            coeffs = coeffs * norm_elmt

        # combine the kinetics and coulomb elements
        self.kin_coul_elmts = self.kin_elmts + self.coul_elmts.sum(dim=0) # (nelmtstot, nelmtstot)

        # prepare the contracted coefficients and indices
        self.coeffs2 = coeffs * coeffs.unsqueeze(1) # (nelmtstot, nelmtstot)
        if isinstance(self.nelmts, torch.Tensor):
            self.csnelmts = torch.cumsum(self.nelmts, dim=0)-1

        # get the contracted part
        self.kin_coul_mat = self._contract(self.kin_coul_elmts) # (nbasis, nbasis)
        self.olp_mat = self._contract(self.olp_elmts) # (nbasis, nbasis)

        # get the normalization constant
        norm = 1. / torch.sqrt(self.olp_mat.diagonal(dim1=-2, dim2=-1)) # (nbasis)
        norm_mat = norm.unsqueeze(-1) * norm.unsqueeze(1) # (nbasis, nbasis)

        # normalize the contracted matrix
        self.kin_coul_mat = self.kin_coul_mat * norm_mat
        self.olp_mat = self.olp_mat * norm_mat

        # make sure the matrix is symmetric
        self.kin_coul_mat = (self.kin_coul_mat + self.kin_coul_mat.transpose(-2,-1)) * 0.5
        self.olp_mat = (self.olp_mat + self.olp_mat.transpose(-2,-1)) * 0.5

        # basis
        self.basis = None
        self.grad_basis = None

        # parameters for set_basis
        self.centres = centres  # (nelmtstot, 3)
        self.ijks = ijks  # (nelmtstot, 3)
        self.coeffs = coeffs  # (nelmtstot,)
        self.alphas = alphas  # (nelmtstot,)
        self.norm = norm  # (nbasis,)

    def set_basis(self, gradlevel=0):
        assert gradlevel >= 0 and gradlevel <= 2
        eps = 1e-15
        dtype = self.norm.dtype
        device = self.norm.device

        # setup the basis
        self.rgrid = self.grid.rgrid_in_xyz # (nr, 3)
        rab = self.rgrid - self.centres.unsqueeze(1) # (nbasis*nelmts, nr, 3)
        dist_sq = torch.einsum("...d,...d->...", rab, rab)
        rab_power = ((rab + eps)**self.ijks.unsqueeze(1)).prod(dim=-1) # (nbasis*nelmts, nr)
        exp_factor = torch.exp(-self.alphas.unsqueeze(-1) * dist_sq)  # (nelmtstot, nr)
        basis_all = rab_power * exp_factor  # (nbasis*nelmts, nr)
        basis_all_coeff = basis_all * self.coeffs.unsqueeze(-1) # (nelmtstot, nr)
        # contract the basis
        basis = self._contract_basis(basis_all_coeff)  # (nbasis, nr)
        norm_basis = basis * self.norm.squeeze(0).unsqueeze(-1)
        self.basis = norm_basis # (nbasis, nr)
        self.basis_dvolume = self.basis * self.grid.get_dvolume() # (nbasis, nr)

        if gradlevel == 0:
            return

        # setup the gradient of the basis
        grad_basis = (basis_all * (-2 * self.alphas.unsqueeze(-1))).unsqueeze(-1)
        grad_basis = grad_basis * rab  # (nelmtstot, nr, 3)
        # (nelmtstot, 3, 3)
        ijks_min_1 = self.ijks.unsqueeze(-1) - torch.eye(3, dtype=dtype, device=device)
        grad_basis2 = ((rab.unsqueeze(-1) + eps)**(ijks_min_1.unsqueeze(1))).prod(dim=-2)  # (nelmtstot, nr, 3)
        grad_basis2 *= self.ijks.unsqueeze(1)
        grad_basis2 *= exp_factor.unsqueeze(-1)
        grad_basis += grad_basis2  # (nelmtstot, nr, 3)
        grad_basis *= self.coeffs.unsqueeze(-1).unsqueeze(-1)  # (nelmtstot, nr, 3)
        grad_basis = self._contract_basis(grad_basis)  # (nbasis, nr, 3)
        grad_basis_norm = grad_basis * self.norm.unsqueeze(-1).unsqueeze(-1)
        self.grad_basis = grad_basis_norm  # (nbasis, nr, 3)

    def _contract_basis(self, basis_inp):
        basis_shape = basis_inp.shape[1:]
        basis_inp = basis_inp.view(basis_inp.shape[0], -1)

        if isinstance(self.nelmts, torch.Tensor):
            basis = basis_inp.cumsum(dim=0)[self.csnelmts,:] # (nbasis, nr)
            basis = torch.cat((basis[:1,:], basis[1:,:]-basis[:-1,:]), dim=0) # (nbasis, nr)
        else:
            basis = basis_inp.view(self.nbasis, self.nelmts, -1).sum(dim=1) # (nbasis, nr)

        return basis.view(-1, *basis_shape)

    def get_kincoul(self):
        # kin_coul_mat: (nbasis, nbasis)
        return xt.LinearOperator.m(self.kin_coul_mat, is_hermitian=True)

    def get_vext(self, vext):
        # vext: (..., nr)
        if self.basis is None:
            raise RuntimeError("Please call `set_basis(gradlevel>=0)` to call this function")
        mat = torch.einsum("...r,br,cr->...bc", vext, self.basis_dvolume, self.basis)
        mat = (mat + mat.transpose(-2,-1)) * 0.5 # ensure the symmetricity
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def get_grad_vext(self, grad_vext):
        # grad_vext: (..., nr, 3)
        if self.grad_basis is None:
            raise RuntimeError("Please call `set_basis(gradlevel>=1)` to call this function")
        mat = torch.einsum("...rd,br,crd->...bc", grad_vext, self.basis_dvolume, self.grad_basis)
        mat = mat + mat.transpose(-2, -1)  # Martin, et. al., eq. (8.14)
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def get_overlap(self):
        return xt.LinearOperator.m(self.olp_mat, is_hermitian=True)

    def dm2dens(self, dm, calc_gradn=False):
        # dm: (*BD, nbasis, nbasis)
        # self.basis: (nbasis, nr)
        # return: (*BD, nr), (*BD, nr, 3)
        dens = torch.einsum("...ij,ir,jr->...r", dm, self.basis, self.basis)

        # calculate the density gradient
        gdens = None
        if calc_gradn:
            if self.grad_basis is None:
                raise RuntimeError("Please call `set_basis(gradlevel>=1)` to calculate the density gradient")
            # (*BD, nr, 3)
            gdens = torch.einsum("...ij,ird,jr->...rd", 2 * dm, self.grad_basis, self.basis)

        res = DensityInfo(density=dens, gradn=gdens)
        return res

    def getparamnames(self, methodname, prefix=""):
        if methodname == "get_kincoul":
            return [prefix+"kin_coul_mat"]
        elif methodname == "get_vext":
            return [prefix+"basis_dvolume", prefix+"basis"]
        elif methodname == "get_overlap":
            return [prefix+"olp_mat"]
        elif methodname == "dm2dens":
            return [prefix+"basis"]
        else:
            raise KeyError("getparamnames has no %s method" % methodname)

    ############################# helper functions #############################
    def _contract(self, mat):
        # multiply the matrix with the contracted coefficients
        mat = mat * self.coeffs2

        # resize mat to have shape of (-1, nbasis, nelmts, nbasis, nelmts)
        batch_size = mat.shape[:-2]
        mat_size = mat.shape[-2:]
        if isinstance(self.nelmts, torch.Tensor):
            mat = mat.view(-1, *mat_size) # (-1, nelmtstot, nelmtstot)
            csmat1 = torch.cumsum(mat, dim=1)[:,self.csnelmts,:] # (-1, nbasis, nelmtstot)
            mat1 = torch.cat((csmat1[:,:1,:], csmat1[:,1:,:]-csmat1[:,:-1,:]), dim=1) # (-1, nbasis, nelmtstot)
            csmat2 = torch.cumsum(mat1, dim=2)[:,:,self.csnelmts] # (-1, nbasis, nbasis)
            cmat = torch.cat((csmat2[:,:,:1], csmat2[:,:,1:]-csmat2[:,:,:-1]), dim=2) # (-1, nbasis, nbasis)
        else:
            mat = mat.view(-1, self.nbasis, self.nelmts, self.nbasis, self.nelmts)

            # sum the nelmts to get the basis
            cmat = mat.sum(dim=-1).sum(dim=-2) # (-1, nbasis, nbasis)
        cmat = cmat.view(*batch_size, self.nbasis, self.nbasis)
        return cmat

class Ecoeff(object):
    # ref: https://joshuagoings.com/2017/04/28/integrals/ and
    # Helgaker, Trygve, and Peter R. Taylor. “Gaussian basis sets and molecular integrals.” Modern Electronic Structure (1995).
    def __init__(self, ijks, centres, alphas, atompos):
        # ijks: (nbasis*nelmts, 3)
        # centres: (nbasis*nelmts, 3)
        # alphas: (nbasis*nelmts)
        # atompos: (natoms, 3)
        self.max_basis = 8
        self.max_ijkflat = 64
        ijk_left = ijks.unsqueeze(0) # (1, nbasis*nelmts, 3)
        ijk_right = ijks.unsqueeze(1) # (nbasis*nelmts, 1, 3)
        self.ndim = ijk_left.shape[-1]
        ijk_pairs = ijk_left * self.max_basis + ijk_right # (nbasis*nelmts, nbasis*nelmts, 3)
        ijk_pairs2 = ijk_pairs[:,:,0] * self.max_ijkflat*self.max_ijkflat + \
                     ijk_pairs[:,:,1] * self.max_ijkflat + ijk_pairs[:,:,2] # (nbasis*nelmts, nbasis*nelmts)
        qab = centres - centres.unsqueeze(1) # (nbasis*nelmts, nbasis*nelmts, 3)
        qab_sq = qab**2 # (nbasis*nelmts, nbasis*nelmts, 3)
        gamma = (alphas + alphas.unsqueeze(1)) # (nbasis*nelmts, nbasis*nelmts)
        kappa = (alphas * alphas.unsqueeze(1)) / gamma # (nbasis*nelmts, nbasis*nelmts)
        mab = torch.exp(-kappa.unsqueeze(-1) * qab_sq) # (nbasis*nelmts, nbasis*nelmts, 3)
        ra = alphas.unsqueeze(-1) * centres # (nbasis*nelmts, 3)
        rc = (ra + ra.unsqueeze(1)) / gamma.unsqueeze(-1) # (nbasis*nelmts, nbasis*nelmts, 3)
        rcd = rc - atompos.unsqueeze(1).unsqueeze(1) # (natoms, nbasis*nelmts, nbasis*neltms, 3)
        rcd_sq = (rcd*rcd).sum(dim=-1) # (natoms, nbasis*nelmts, nbasis*neltms)

        self.ijk_left = ijk_left
        self.ijk_right = ijk_right
        self.ijk_pairs = ijk_pairs
        self.ijk_pairs2 = ijk_pairs2
        self.alphas = alphas.unsqueeze(0)
        self.betas = alphas.unsqueeze(1)
        self.qab = qab
        self.qab_sq = qab_sq
        self.gamma = gamma
        self.kappa = kappa
        self.mab = mab
        self.ra = ra
        self.rc = rc
        self.rcd = rcd
        self.rcd_sq = rcd_sq

        self.ijk_left_max = self.ijk_left.max()
        self.ijk_right_max = self.ijk_right.max()
        self.ijk_pairs2_unique = torch.unique(self.ijk_pairs2)
        self.idx_ijk = [(self.ijk_pairs2 == ijk_flat_value) for ijk_flat_value in self.ijk_pairs2_unique]

        # the key is: "i,j,t,xyz"
        self.key_format = "{},{},{},{}"
        self.rkey_format = "{},{},{},{}"
        # the value's shape is: (nbasis*nelmts, nbasis*nelmts)
        self.e_memory = {
            "0,0,0,0": self.mab[:,:,0],
            "0,0,0,1": self.mab[:,:,1],
            "0,0,0,2": self.mab[:,:,2],
        }
        # the key format is: r,s,t,n
        # the value's shape is: (natoms, nbasis*nelmts, nbasis*nelmts)
        self.r_memory = {}

        # cache
        self.overlap = None
        self.kinetics = None
        self.coulomb = None

    def get_overlap(self):
        if self.overlap is None:
            # NOTE: using this makes test_grad_intermediate.py::test_grad_dft_cgto produces nan in gradgradcheck energy
            # (nans are in the numerical, not in the analytic one)
            self.overlap = get_overlap_mat(self.ijk_left_max, self.ijk_right_max,
                self.max_basis, self.ndim, self.ijk_pairs,
                self.alphas, self.betas, self.gamma, self.kappa, self.qab,
                self.e_memory, self.key_format)

            # # python code for profiling with pprofile
            # overlap_dim = torch.empty_like(self.qab); # (nbasis_tot, nbasis_tot, ndim)
            # for i in range(self.ijk_left_max + 1):
            #     for j in range(self.ijk_right_max + 1):
            #         idx = (self.ijk_pairs == (i * self.max_basis + j)); # (nbasis_tot, nbasis_tot, ndim)
            #         for xyz in range(self.ndim):
            #             idxx = idx[..., xyz]
            #             coeff = self.get_ecoeff(i, j, 0, xyz)
            #             overlap_dim[..., xyz][idxx] = coeff[idxx]
            # res = overlap_dim.prod(dim=-1) * torch.pow(np.pi / self.gamma, 1.5); # (nbasis_tot, nbasis_tot)
            # self.overlap = res

        return self.overlap

    def get_kinetics(self):
        if self.kinetics is None:
            self.kinetics = get_kinetics_mat(self.ijk_left_max, self.ijk_right_max,
                self.max_basis, self.ndim, self.ijk_pairs,
                self.alphas, self.betas, self.gamma, self.kappa, self.qab,
                self.e_memory, self.key_format)

            # # python code for profiling with pprofile
            # kinetics_dim0 = torch.empty_like(self.qab) # (nbasis_tot, nbasis_tot, ndim)
            # kinetics_dim1 = torch.empty_like(self.qab) # (nbasis_tot, nbasis_tot, ndim)
            # kinetics_dim2 = torch.empty_like(self.qab) # (nbasis_tot, nbasis_tot, ndim)
            # for i in range(self.ijk_left_max + 1):
            #     for j in range(self.ijk_right_max + 1):
            #         idx = (self.ijk_pairs == (i * self.max_basis + j))
            #         for xyz in range(self.ndim):
            #             idxx = idx[..., xyz]
            #             sij = self.get_ecoeff(i, j  , 0, xyz)
            #             d1  = self.get_ecoeff(i, j-2, 0, xyz)
            #             d2  = self.get_ecoeff(i, j+2, 0, xyz)
            #             dij = j*(j-1)*d1 - 2*(2*j+1)*self.betas*sij + 4*self.betas*self.betas*d2;
            #             sij_idxx = sij[idxx]
            #             dij_idxx = dij[idxx]
            #             if xyz == 0:
            #                 kinetics_dim0[..., xyz][idxx] = dij_idxx
            #                 kinetics_dim1[..., xyz][idxx] = sij_idxx
            #                 kinetics_dim2[..., xyz][idxx] = sij_idxx
            #             elif xyz == 1:
            #                 kinetics_dim0[..., xyz][idxx] = sij_idxx
            #                 kinetics_dim1[..., xyz][idxx] = dij_idxx
            #                 kinetics_dim2[..., xyz][idxx] = sij_idxx
            #             else:
            #                 kinetics_dim0[..., xyz][idxx] = sij_idxx
            #                 kinetics_dim1[..., xyz][idxx] = sij_idxx
            #                 kinetics_dim2[..., xyz][idxx] = dij_idxx
            # kinetics = kinetics_dim0.prod(dim=-1) + kinetics_dim1.prod(dim=-1) + kinetics_dim2.prod(dim=-1)
            # res = -0.5 * torch.pow(np.pi / self.gamma, 1.5) * kinetics
            # self.kinetics = res

        return self.kinetics

    def get_coulomb(self):
        # returns (natoms, nbasis*nelmts, nbasis*nelmts)
        if self.coulomb is None:
            self.coulomb = get_coulomb_mat(
                self.max_ijkflat, self.max_basis,
                self.idx_ijk,
                self.rcd_sq,
                self.ijk_pairs2_unique,
                self.alphas, self.betas, self.gamma, self.kappa, self.qab,
                self.e_memory, self.key_format,
                self.rcd, self.r_memory, self.rkey_format)

            # # python code for profiling with pprofile
            # coulomb = torch.zeros_like(self.rcd_sq)
            # numel = self.ijk_pairs2_unique.numel()
            # max_ijkflat = self.max_ijkflat
            # max_basis = self.max_basis
            # for i in range(numel):
            #     ijk_flat_value = int(self.ijk_pairs2_unique[i])
            #     idx = self.idx_ijk[i]
            #     rcd2 = self.rcd
            #     rcd_sq2 = self.rcd_sq
            #     gamma2 = self.gamma
            #
            #     ijk_pair2 = ijk_flat_value % max_ijkflat
            #     ijk_pair1 = (ijk_flat_value // max_ijkflat) % max_ijkflat
            #     ijk_pair0 = (ijk_flat_value // max_ijkflat) // max_ijkflat
            #     k = ijk_pair0 // max_basis
            #     l = ijk_pair1 // max_basis
            #     m = ijk_pair2 // max_basis
            #     u = ijk_pair0 % max_basis
            #     v = ijk_pair1 % max_basis
            #     w = ijk_pair2 % max_basis
            #
            #     for r in range(k + u + 1):
            #         Erku = self.get_ecoeff(k, u, r, 0)[idx]
            #         for s in range(l + v + 1):
            #             Eslv = self.get_ecoeff(l, v, s, 1)[idx]
            #             for t in range(m + w + 1):
            #                 Etmw = self.get_ecoeff(m, w, t, 2)[idx]
            #                 Rrst = self.get_rcoeff(r, s, t, 0,
            #                     rcd2, rcd_sq2, gamma2)[:,idx]
            #                 coulomb[:,idx] += (Erku * Eslv * Etmw) * Rrst
            #
            # coulomb *= -(2 * np.pi / self.gamma)
            # self.coulomb = coulomb
        return self.coulomb

    # python code for profiling only
    def get_ecoeff(self, i, j, t, xyz):
        if (t < 0) or (t > (i + j)) or (i < 0) or (j < 0):
            return torch.zeros_like(self.qab[..., 0])
        key = self.key_format.format(i, j, t, xyz)
        if key in self.e_memory:
            return self.e_memory[key]

        if (i == 0) and (j > 0):
            c1 = self.get_ecoeff(i, j-1, t-1, xyz)
            c2 = self.get_ecoeff(i, j-1, t  , xyz)
            c3 = self.get_ecoeff(i, j-1, t+1, xyz)
            coeff = 1. / (2 * self.gamma) * c1 + self.kappa * self.qab[..., xyz] / self.betas * c2 + (t + 1) * c3
        else:
            c1 = self.get_ecoeff(i-1, j, t-1, xyz)
            c2 = self.get_ecoeff(i-1, j, t  , xyz)
            c3 = self.get_ecoeff(i-1, j, t+1, xyz)
            coeff = 1. / (2 * self.gamma) * c1 - self.kappa * self.qab[..., xyz] / self.alphas * c2 + (t + 1) * c3
        self.e_memory[key] = coeff
        return coeff

    # python code for profiling only
    def get_rcoeff(self, r, s, t, n, rcd, rcd_sq, gamma):
        if r < 0 or s < 0 or t < 0:
            return 0

        key = self.rkey_format.format(r, s, t, n)
        if key in self.r_memory:
            return self.r_memory[key]

        if r == 0 and s == 0 and t == 0:
            gamma_rcd = gamma * rcd_sq
            coeff = (-2 * gamma)**n * self.boys(n, gamma_rcd)
        elif r > 0:
            c1 = self.get_rcoeff(r-2, s, t, n+1, rcd, rcd_sq, gamma)
            c2 = self.get_rcoeff(r-1, s, t, n+1, rcd, rcd_sq, gamma)
            coeff = (r-1) * c1 + rcd[..., 0] * c2
        elif s > 0:
            c1 = self.get_rcoeff(r, s-2, t, n+1, rcd, rcd_sq, gamma)
            c2 = self.get_rcoeff(r, s-1, t, n+1, rcd, rcd_sq, gamma)
            coeff = (s-1) * c1 + rcd[..., 1] * c2
        else:
            c1 = self.get_rcoeff(r, s, t-2, n+1, rcd, rcd_sq, gamma)
            c2 = self.get_rcoeff(r, s, t-1, n+1, rcd, rcd_sq, gamma)
            coeff = (t-1) * c1 + rcd[..., 2] * c2
        self.r_memory[key] = coeff
        return coeff

    def boys(self, n, t):
        nhalf = torch.tensor(n + 0.5, dtype=t.dtype, device=t.device)
        t2 = t + 1e-12
        exp_part = -nhalf * torch.log(t2) + torch.lgamma(nhalf)
        return 0.5 * torch.igamma(nhalf, t2) * torch.exp(exp_part)
