from typing import List
import torch
import xitorch as xt
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.hamilton.libcint_wrapper import LibcintWrapper
from dqc.utils.datastruct import AtomCGTOBasis, ValGrad
from dqc.grid.base_grid import BaseGrid
from dqc.xc.base_xc import BaseXC

class HMolCGTO(BaseHamilton):
    def __init__(self, atombases: List[AtomCGTOBasis], spherical: bool = True) -> None:
        self.atombases = atombases
        self.libcint_wrapper = LibcintWrapper(atombases, spherical)
        self.dtype = self.libcint_wrapper.dtype
        self.device = self.libcint_wrapper.device

        # get the matrices (all (nao, nao), except el_mat)
        # these matrices have already been normalized
        self.olp_mat = self.libcint_wrapper.overlap()
        kin_mat = self.libcint_wrapper.kinetic()
        nucl_mat = self.libcint_wrapper.nuclattr()
        self.kinnucl_mat = kin_mat + nucl_mat
        self.el_mat = self.libcint_wrapper.elrep()  # (nao^4)

        self.is_grid_set = False
        self.is_ao_set = False
        self.is_grad_ao_set = False
        self.is_lapl_ao_set = False

    def get_kinnucl(self) -> xt.LinearOperator:
        # kinnucl_mat: (nao, nao)
        # return: (nao, nao)
        return xt.LinearOperator.m(self.kinnucl_mat, is_hermitian=True)

    def get_overlap(self) -> xt.LinearOperator:
        # olp_mat: (nao, nao)
        # return: (nao, nao)
        return xt.LinearOperator.m(self.olp_mat, is_hermitian=True)

    def get_elrep(self, dm: torch.Tensor) -> xt.LinearOperator:
        # dm: (*BD, nao, nao)
        # elrep_mat: (nao, nao, nao, nao)
        # return: (*BD, nao, nao)
        mat = torch.einsum("...ij,ijkl->...kl", dm, self.el_mat)
        mat = (mat + mat.transpose(-2, -1)) * 0.5  # reduce numerical instability
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def ao_orb2dm(self, orb: torch.Tensor, orb_weight: torch.Tensor) -> torch.Tensor:
        # convert the atomic orbital to the density matrix
        # in CGTO, it is U.W.U^T

        # orb: (*BO, norb, nao)
        # orb_weight: (*BW, norb)
        # return: (*BOW, nao, nao)

        orb_w = orb * orb_weight.unsqueeze(-1)  # (*BOW, norb, nao)
        return torch.matmul(orb_w.transpose(-2, -1), orb)  # (*BOW, nao, nao)

    ############### grid-related ###############
    def setup_grid(self, grid: BaseGrid, xcfamily: int = 0) -> None:
        # save the grid
        self.grid = grid
        self.rgrid = grid.get_rgrid()
        assert grid.coord_type == "cart"

        # setup the basis as a spatial function
        self.is_ao_set = True
        self.basis = self.libcint_wrapper.eval_gto(self.rgrid)  # (nao, ngrid)
        self.basis_dvolume = self.basis * self.grid.get_dvolume()  # (nao, ngrid)

        if xcfamily == 1:  # LDA
            return

        # setup the gradient of the basis
        self.is_grad_ao_set = True
        self.grad_basis = self.libcint_wrapper.eval_gradgto(self.rgrid)  # (ndim, nao, ngrid)
        if xcfamily == 2:  # GGA
            return

        # setup the laplacian of the basis
        self.is_lapl_ao_set = True
        self.lapl_basis = self.libcint_wrapper.eval_laplgto(self.rgrid)  # (nao, ngrid)

    def get_vext(self, vext: torch.Tensor) -> xt.LinearOperator:
        # vext: (*BR, ngrid)
        if not self.is_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, gradlevel>=0)` to call this function")
        mat = torch.einsum("...r,br,cr->...bc", vext, self.basis_dvolume, self.basis)  # (*BR, nao, nao)
        mat = (mat + mat.transpose(-2, -1)) * 0.5  # ensure the symmetricity and reduce numerical instability
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def get_grad_vext(self, grad_vext: torch.Tensor) -> xt.LinearOperator:
        # grad_vext: (*BR, ngrid, ndim)
        if not self.is_grad_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, gradlevel>=1)` to call this function")
        mat = torch.einsum("...rd,br,dcr->...bc", grad_vext, self.basis_dvolume, self.grad_basis)
        mat = mat + mat.transpose(-2, -1)  # Martin, et. al., eq. (8.14)
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def get_lapl_vext(self, lapl_vext: torch.Tensor) -> xt.LinearOperator:
        # get the linear operator for the laplacian part of the potential
        # lapl_vext: (*BR, ngrid)
        # return: (*BR, nao, nao)
        # TODO: implement this!
        pass

    ################ xc-related ################
    def get_vxc(self, xc: BaseXC, dm: torch.Tensor) -> xt.LinearOperator:
        # dm: (*BD, nao, nao)
        densinfo = self._dm2densinfo(dm, xc.family)  # value: (*BD, nr)
        potinfo = xc.get_vxc(densinfo)  # value: (*BD, nr)

        # get the linear operator from the potential
        vxc_linop = self.get_vext(potinfo.value)
        if potinfo.grad is not None:
            vxc_linop = vxc_linop + self.get_grad_vext(potinfo.grad)
        if potinfo.lapl is not None:
            vxc_linop = vxc_linop + self.get_lapl_vext(potinfo.lapl)

        return vxc_linop

    def _dm2densinfo(self, dm: torch.Tensor, family: int) -> ValGrad:
        # dm: (*BD, nao, nao)
        # family: 1 for LDA, 2 for GGA, 3 for MGGA
        # self.basis: (nao, ngrid)
        dens = torch.einsum("...ij,ir,jr->...r", dm, self.basis, self.basis)

        # calculate the density gradient
        gdens = None
        if family >= 2:  # requires gradient
            if not self.is_grad_ao_set:
                raise RuntimeError("Please call `setup_grid(grid, gradlevel>=1)` to calculate the density gradient")
            # (*BD, ngrid, ndim)
            gdens = torch.einsum("...ij,dir,jr->...rd", dm, self.grad_basis, self.basis)

        # TODO: implement the density laplacian

        # dens: (*BD, ngrid)
        # gdens: (*BD, ngrid, ndim)
        res = ValGrad(value=dens, grad=gdens)
        return res

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_kinnucl":
            return [prefix + "kinnucl_mat"]
        elif methodname == "get_overlap":
            return [prefix + "olp_mat"]
        elif methodname == "get_elrep":
            return [prefix + "elrep_mat"]
        elif methodname == "ao_orb2dm":
            return []
        elif methodname == "get_vext":
            return [prefix + "basis_dvolume", prefix + "basis"]
        elif methodname == "get_grad_vext":
            return [prefix + "basis_dvolume", prefix + "grad_basis"]
        elif methodname == "get_lapl_vext":
            return [prefix + "basis_dvolume", prefix + "lapl_basis"]
        else:
            raise KeyError("getparamnames has no %s method" % methodname)
        # TODO: complete this
