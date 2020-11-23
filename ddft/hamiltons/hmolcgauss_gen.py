from abc import abstractmethod, abstractproperty
from functools import reduce
import torch
import numpy as np
import xitorch as xt

from ddft.hamiltons.base_hamilton_gen import BaseHamiltonGenerator, DensityInfo
from ddft.utils.spharmonics import spharmonics
from ddft.utils.gamma import incgamma
# from ddft.csrc import get_ecoeff, get_overlap_mat, get_kinetics_mat, \
#     get_coulomb_mat
from ddft.integrals.cgauss import overlap, kinetic, nuclattr, elrep

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
        with_elrep = True  # set to False to disable electron repulsion

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
        ecoeff_obj = Ecoeff2(ijks, centres, alphas, atompos)
        self.olp_elmts = ecoeff_obj.get_overlap() # (nelmtstot, nelmtstot)
        self.kin_elmts = ecoeff_obj.get_kinetics() # (nelmtstot, nelmtstot)
        self.coul_elmts = ecoeff_obj.get_nuclattr() * atomzs.unsqueeze(-1).unsqueeze(-1) # (natoms, nelmtstot, nelmtstot)
        if with_elrep:
            self.elrep_elmts = ecoeff_obj.get_elrep()  # (nelmtstot, nelmtstot, nelmtstot, nelmtstot)

        # normalize each gaussian elements
        if normalize_elmts:
            norm_elmt = 1./torch.sqrt(torch.diagonal(self.olp_elmts)) # (nelmtstot,)
            coeffs = coeffs * norm_elmt

        # combine the kinetics and coulomb elements
        self.kin_coul_elmts = self.kin_elmts + self.coul_elmts.sum(dim=0) # (nelmtstot, nelmtstot)

        # prepare the contracted coefficients and indices
        self.coeffs2 = coeffs * coeffs.unsqueeze(1) # (nelmtstot, nelmtstot)
        if with_elrep:
            self.coeffs4 = self.coeffs2[:, :, None, None] * self.coeffs2  # (nelmtstot^4)
        if isinstance(self.nelmts, torch.Tensor):
            self.csnelmts = torch.cumsum(self.nelmts, dim=0) - 1

        # get the contracted part
        self.kin_coul_mat = self._contract(self.kin_coul_elmts) # (nbasis, nbasis)
        self.olp_mat = self._contract(self.olp_elmts) # (nbasis, nbasis)
        if with_elrep:
            self.elrep_mat = self._contract_elrep(self.elrep_elmts)  # (nbasis^4)

        # get the normalization constant
        norm = 1. / torch.sqrt(self.olp_mat.diagonal(dim1=-2, dim2=-1)) # (nbasis)
        norm_mat = norm.unsqueeze(-1) * norm.unsqueeze(1) # (nbasis, nbasis)

        # normalize the contracted matrix
        self.kin_coul_mat = self.kin_coul_mat * norm_mat
        self.olp_mat = self.olp_mat * norm_mat
        if with_elrep:
            self.elrep_mat = self.elrep_mat * (norm_mat[:, :, None, None] * norm_mat)

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

    def get_elrep(self, dm):
        # dm: (*BD, nbasis, nbasis)
        # elrep_mat: (nbasis, nbasis, nbasis, nbasis)
        mat = torch.einsum("...ij,ijkl->...kl", dm, self.elrep_mat)
        mat = (mat + mat.transpose(-2, -1)) * 0.5
        return xt.LinearOperator.m(mat, is_hermitian=True)

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
        elif methodname == "get_elrep":
            return [prefix+"elrep_mat"]
        elif methodname == "dm2dens":
            return [prefix+"basis"]
        else:
            raise KeyError("getparamnames has no %s method" % methodname)

    ############################# helper functions #############################
    def _contract(self, mat):
        # mat: (*BM, nelmtstot^2)
        # multiply the matrix with the contracted coefficients
        mat = mat * self.coeffs2

        # resize mat to have shape of (-1, nbasis, nelmts, nbasis, nelmts)
        batch_size = mat.shape[:-2]
        mat_size = mat.shape[-2:]
        if isinstance(self.nelmts, torch.Tensor):
            mat = mat.view(-1, *mat_size) # (-1, nelmtstot, nelmtstot)
            mat = contract_dim(mat, dim=1, csnelmts=self.csnelmts)
            cmat = contract_dim(mat, dim=2, csnelmts=self.csnelmts)
        else:
            # resize mat to have shape of (-1, nbasis, nelmts, nbasis, nelmts)
            mat = mat.view(-1, self.nbasis, self.nelmts, self.nbasis, self.nelmts)

            # sum the nelmts to get the basis
            cmat = mat.sum(dim=-1).sum(dim=-2) # (-1, nbasis, nbasis)
        cmat = cmat.view(*batch_size, self.nbasis, self.nbasis)
        return cmat

    def _contract_elrep(self, mat):
        # mat: (*BM, nelmtstot^4)
        # multiply the matrix with the contracted coefficients
        mat = mat * self.coeffs4

        batch_size = mat.shape[:-4]
        mat_size = mat.shape[-4:]
        if isinstance(self.nelmts, torch.Tensor):
            cmat = mat.view(-1, *mat_size) # (-1, nelmtstot^4)
            cmat = contract_dim(cmat, dim=1, csnelmts=self.csnelmts)
            cmat = contract_dim(cmat, dim=2, csnelmts=self.csnelmts)
            cmat = contract_dim(cmat, dim=3, csnelmts=self.csnelmts)
            cmat = contract_dim(cmat, dim=4, csnelmts=self.csnelmts)  # (-1, nbasis^4)
        else:
            # resize mat to have shape of (-1, nbasis, nelmts, nbasis, nelmts, nbasis, nelmts, nbasis, nelmts)
            mat = mat.view(-1, self.nbasis, self.nelmts, self.nbasis, self.nelmts,
                               self.nbasis, self.nelmts, self.nbasis, self.nelmts)

            # sum the nelmts to get the basis
            cmat = mat.sum(dim=-1).sum(dim=-2).sum(dim=-3).sum(dim=-4) # (-1, nbasis^4)

        cmat = cmat.view(*batch_size, self.nbasis, self.nbasis, self.nbasis, self.nbasis)
        return cmat

def contract_dim(mat, dim, csnelmts):
    # contracting the matrix at the dimension dim
    # mat: (..., nelmtstot, ...)
    # res: (..., nbasis, ...)
    if dim < 0:
        dim = dim + mat.ndim
    index = (slice(None, None, None), ) * dim
    csmat = torch.cumsum(mat, dim=dim)[index + (csnelmts,)]
    resmat = torch.cat(
        (csmat[index + (slice(None, 1, None), )],
         csmat[index + (slice(1, None, None), )] - csmat[index + (slice(None, -1, None), )]),
        dim=dim)
    return resmat

class Ecoeff2(object):
    def __init__(self, ijks, centres, alphas, atompos):
        # ijks: (nbasis*nelmts, 3)
        # centres: (nbasis*nelmts, 3)
        # alphas: (nbasis*nelmts)
        # atompos: (natoms, 3)

        nelmtstot = alphas.shape[0]
        natoms = atompos.shape[0]
        ijks = ijks.transpose(-2, -1)  # (3, nelmtstot)
        centres = centres.transpose(-2, -1)  # (3, nelmtstot)
        atompos = atompos.transpose(-2, -1)  # (3, natoms)

        dtype = alphas.dtype
        device = alphas.device
        self.zeros_natoms = torch.zeros((natoms, nelmtstot), dtype=dtype, device=device)

        self.a = alphas  # (nelmtstot,)
        self.pos = centres  # (3, nelmtstot)
        self.lmn = ijks  # (3, nelmtstot)
        self.atompos = atompos  # (3, natoms)

    def get_overlap(self):
        return overlap(self.a, self.pos, self.lmn, self.a, self.pos, self.lmn)

    def get_kinetics(self):
        return kinetic(self.a, self.pos, self.lmn, self.a, self.pos, self.lmn)

    def get_nuclattr(self):
        return nuclattr(self.a, self.pos, self.lmn, self.a, self.pos, self.lmn, self.atompos)

    def get_elrep(self):
        return elrep(self.a, self.pos, self.lmn,
                     self.a, self.pos, self.lmn,
                     self.a, self.pos, self.lmn,
                     self.a, self.pos, self.lmn)
