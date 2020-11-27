import torch
from typing import List
from ddft.hamiltons.base_hamilton_gen import BaseHamiltonGenerator
from ddft.hamiltons.libcintwrapper import LibcintWrapper
from ddft.basissets.cgtobasis import AtomCGTOBasis
from ddft.grids.base_grid import Base3DGrid

class HMolCGTO(BaseHamiltonGenerator):
    def __init__(self, grid: Base3DGrid, atombases: List[AtomCGTOBasis],
                 spherical: bool = True) -> None:
        self.atombases = atombases
        self.libcint_wrapper = LibcintWrapper(atombases, spherical)
        nb = self.libcint_wrapper.nbases_tot
        shape = (nb, nb)
        dtype = self.libcint_wrapper.dtype
        device = self.libcint_wrapper.device
        super().__init__(grid, shape, dtype, device)

        # get the matrices (all (nao, nao), except el_mat)
        # these matrices have already been normalized
        self.olp_mat = self.libcint_wrapper.overlap()
        kin_mat = self.libcint_wrapper.kinetic()
        nucl_mat = self.libcint_wrapper.nuclattr()
        self.kinnucl_mat = kin_mat + nucl_mat
        self.el_mat = self.libcint_wrapper.elrep()  # (nao^4)

        self.rgrid = grid.rgrid_in_xyz  # (ngrid, ndim)
        self.is_ao_set = False
        self.is_grad_ao_set = False
        self.is_lapl_ao_set = False

    def set_basis(self, gradlevel: int = 0) -> None:
        assert gradlevel >= 0 and gradlevel <= 2

        # setup the basis
        self.is_ao_set = True
        self.basis = self.libcint_wrapper.eval_gto(self.rgrid)  # (nao, ngrid)
        self.basis_dvolume = self.basis * self.grid.get_dvolume()  # (nao, ngrid)

        if gradlevel == 0:
            return

        self.is_grad_ao_set = True
        self.grad_basis = self.libcint_wrapper.eval_gradgto(self.rgrid)  # (ndim, nao, ngrid)
        if grad_level == 1:
            return

        self.is_lapl_ao_set = True
        self.lapl_basis = self.libcint_wrapper.eval_laplgto(self.rgrid)  # (nao, ngrid)

    def get_kincoul(self):
        # kin_coul_mat: (nbasis, nbasis)
        return xt.LinearOperator.m(self.kinnucl_mat, is_hermitian=True)

    def get_elrep(self, dm):
        # dm: (*BD, nao, nao)
        # elrep_mat: (nao, nao, nao, nao)
        mat = torch.einsum("...ij,ijkl->...kl", dm, self.el_mat)
        mat = (mat + mat.transpose(-2, -1)) * 0.5
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def get_vext(self, vext):
        # vext: (..., ngrid)
        if not self.is_ao_set:
            raise RuntimeError("Please call `set_basis(gradlevel>=0)` to call this function")
        mat = torch.einsum("...r,br,cr->...bc", vext, self.basis_dvolume, self.basis)
        mat = (mat + mat.transpose(-2,-1)) * 0.5 # ensure the symmetricity
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def get_grad_vext(self, grad_vext):
        # grad_vext: (..., ngrid, ndim)
        if self.is_grad_ao_set is None:
            raise RuntimeError("Please call `set_basis(gradlevel>=1)` to call this function")
        mat = torch.einsum("...rd,br,dcr->...bc", grad_vext, self.basis_dvolume, self.grad_basis)
        mat = mat + mat.transpose(-2, -1)  # Martin, et. al., eq. (8.14)
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def get_overlap(self):
        return xt.LinearOperator.m(self.olp_mat, is_hermitian=True)

    def dm2dens(self, dm, calc_gradn=False):
        # dm: (*BD, nao, nao)
        # self.basis: (nao, ngrid)
        # return: (*BD, ngrid), (*BD, ngrid, 3)
        dens = torch.einsum("...ij,ir,jr->...r", dm, self.basis, self.basis)

        # calculate the density gradient
        gdens = None
        if calc_gradn:
            if self.grad_basis is None:
                raise RuntimeError("Please call `set_basis(gradlevel>=1)` to calculate the density gradient")
            # (*BD, ngrid, ndim)
            gdens = torch.einsum("...ij,dir,jr->...rd", 2 * dm, self.grad_basis, self.basis)

        res = DensityInfo(density=dens, gradn=gdens)
        return res

    def getparamnames(self, methodname, prefix=""):
        if methodname == "get_kincoul":
            return [prefix+"kinnucl_mat"]
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
