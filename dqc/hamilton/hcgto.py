from typing import List, Optional, Union, overload
import torch
import xitorch as xt
import dqc.hamilton.intor as intor
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.utils.datastruct import AtomCGTOBasis, ValGrad, SpinParam
from dqc.grid.base_grid import BaseGrid
from dqc.xc.base_xc import BaseXC

class HamiltonCGTO(BaseHamilton):
    def __init__(self, atombases: List[AtomCGTOBasis], spherical: bool = True) -> None:
        self.atombases = atombases
        self.libcint_wrapper = intor.LibcintWrapper(atombases, spherical)
        self.dtype = self.libcint_wrapper.dtype
        self.device = self.libcint_wrapper.device

        self.is_grid_set = False
        self.is_ao_set = False
        self.is_grad_ao_set = False
        self.is_lapl_ao_set = False
        self.xc: Optional[BaseXC] = None
        self.xcfamily = 1
        self.is_built = False

    @property
    def nao(self) -> int:
        assert self.is_built, "Must run .build() first before calling this function"
        return self.olp_mat.shape[-1]

    def build(self):
        # get the matrices (all (nao, nao), except el_mat)
        # these matrices have already been normalized
        self.olp_mat = intor.overlap(self.libcint_wrapper)
        kin_mat = intor.kinetic(self.libcint_wrapper)
        nucl_mat = intor.nuclattr(self.libcint_wrapper)
        self.kinnucl_mat = kin_mat + nucl_mat
        self.el_mat = intor.elrep(self.libcint_wrapper)  # (nao^4)
        self.is_built = True

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

        # orb: (*BO, nao, norb)
        # orb_weight: (*BW, norb)
        # return: (*BOW, nao, nao)

        orb_w = orb * orb_weight.unsqueeze(-2)  # (*BOW, nao, norb)
        return torch.matmul(orb, orb_w.transpose(-2, -1))  # (*BOW, nao, nao)

    def aodm2dens(self, dm: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (*BR, ndim)
        # dm: (*BD, nao, nao)
        # returns: (*BRD)

        nao = dm.shape[-1]
        xyzshape = xyz.shape
        # basis: (nao, *BR)
        basis = intor.eval_gto(self.libcint_wrapper, xyz.reshape(-1, xyzshape[-1])).reshape((nao, *xyzshape[:-1]))
        basis = torch.movedim(basis, 0, -1)  # (*BR, nao)

        # torch.einsum("...ij,...i,...j->...", dm, basis, basis)
        dens = torch.matmul(dm, basis.unsqueeze(-1))  # (*BRD, nao, 1)
        dens = torch.matmul(basis.unsqueeze(-2), dens).squeeze(-1).squeeze(-1)  # (*BRD)
        return dens

    ############### grid-related ###############
    def setup_grid(self, grid: BaseGrid, xc: Optional[BaseXC] = None) -> None:
        # save the family and save the xc
        self.xc = xc
        if xc is None:
            self.xcfamily = 1
        else:
            self.xcfamily = xc.family

        # save the grid
        self.grid = grid
        self.rgrid = grid.get_rgrid()
        assert grid.coord_type == "cart"

        # setup the basis as a spatial function
        self.is_ao_set = True
        self.basis = intor.eval_gto(self.libcint_wrapper, self.rgrid)  # (nao, ngrid)
        self.basis_dvolume = self.basis * self.grid.get_dvolume()  # (nao, ngrid)

        if self.xcfamily == 1:  # LDA
            return

        # setup the gradient of the basis
        self.is_grad_ao_set = True
        self.grad_basis = intor.eval_gradgto(self.libcint_wrapper, self.rgrid)  # (ndim, nao, ngrid)
        if self.xcfamily == 2:  # GGA
            return

        # setup the laplacian of the basis
        self.is_lapl_ao_set = True
        self.lapl_basis = intor.eval_laplgto(self.libcint_wrapper, self.rgrid)  # (nao, ngrid)

    def get_vext(self, vext: torch.Tensor) -> xt.LinearOperator:
        # vext: (*BR, ngrid)
        if not self.is_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, xc)` to call this function")
        mat = torch.einsum("...r,br,cr->...bc", vext, self.basis_dvolume, self.basis)  # (*BR, nao, nao)
        mat = (mat + mat.transpose(-2, -1)) * 0.5  # ensure the symmetricity and reduce numerical instability
        return xt.LinearOperator.m(mat, is_hermitian=True)

    def get_grad_vext(self, grad_vext: torch.Tensor) -> xt.LinearOperator:
        # grad_vext: (*BR, ngrid, ndim)
        if not self.is_grad_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, xc)` to call this function")
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
    @overload
    def get_vxc(self, dm: SpinParam[torch.Tensor]) -> SpinParam[xt.LinearOperator]:
        ...

    @overload
    def get_vxc(self, dm: torch.Tensor) -> xt.LinearOperator:
        ...

    def get_vxc(self, dm):
        # dm: (*BD, nao, nao)
        assert self.xc is not None, "Please call .setup_grid with the xc object"

        densinfo = self._dm2densinfo(dm, self.xc.family)  # value: (*BD, nr)
        potinfo = self.xc.get_vxc(densinfo)  # value: (*BD, nr)

        if isinstance(dm, torch.Tensor):  # unpolarized case
            # get the linear operator from the potential
            vxc_linop = self.get_vext(potinfo.value)
            if self.xcfamily >= 2:  # GGA or MGGA
                assert potinfo.grad is not None
                vxc_linop = vxc_linop + self.get_grad_vext(potinfo.grad)
            if self.xcfamily >= 3:  # MGGA
                assert potinfo.lapl is not None
                vxc_linop = vxc_linop + self.get_lapl_vext(potinfo.lapl)

            return vxc_linop

        else:  # polarized case
            # get the linear operator from the potential
            vxc_linop_u = self.get_vext(potinfo.u.value)
            vxc_linop_d = self.get_vext(potinfo.d.value)
            if self.xcfamily >= 2:  # GGA or MGGA
                assert potinfo.u.grad is not None
                assert potinfo.d.grad is not None
                vxc_linop_u = vxc_linop_u + self.get_grad_vext(potinfo.u.grad)
                vxc_linop_d = vxc_linop_d + self.get_grad_vext(potinfo.d.grad)
            if self.xcfamily >= 3:  # MGGA
                assert potinfo.u.lapl is not None
                assert potinfo.d.lapl is not None
                vxc_linop_u = vxc_linop_u + self.get_lapl_vext(potinfo.u.lapl)
                vxc_linop_d = vxc_linop_d + self.get_lapl_vext(potinfo.d.lapl)

            return SpinParam(u=vxc_linop_u, d=vxc_linop_d)

    def get_exc(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        assert self.xc is not None, "Please call .setup_grid with the xc object"

        # obtain the energy density per unit volume
        densinfo = self._dm2densinfo(dm, self.xc.family)  # (spin) value: (*BD, nr)
        edens = self.xc.get_edensityxc(densinfo)  # (*BD, nr)

        return torch.sum(self.grid.get_dvolume() * edens, dim=-1)

    @overload
    def _dm2densinfo(self, dm: torch.Tensor, family: int) -> ValGrad:
        ...

    @overload
    def _dm2densinfo(self, dm: SpinParam[torch.Tensor], family: int) -> SpinParam[ValGrad]:
        ...

    def _dm2densinfo(self, dm, family):
        # dm: (*BD, nao, nao)
        # family: 1 for LDA, 2 for GGA, 3 for MGGA
        # self.basis: (nao, ngrid)
        if isinstance(dm, SpinParam):
            res_u = self._dm2densinfo(dm.u, family)
            res_d = self._dm2densinfo(dm.d, family)
            return SpinParam(u=res_u, d=res_d)
        else:
            dens = torch.einsum("...ij,ir,jr->...r", dm, self.basis, self.basis)

            # calculate the density gradient
            gdens = None
            if family >= 2:  # requires gradient
                if not self.is_grad_ao_set:
                    raise RuntimeError("Please call `setup_grid(grid, gradlevel>=1)` to calculate the density gradient")
                # (*BD, ngrid, ndim)
                # dm is multiplied by 2 because n(r) = sum (D_ij * phi_i * phi_j), thus
                # d.n(r) = sum (D_ij * d.phi_i * phi_j + D_ij * phi_i * d.phi_j)
                gdens = torch.einsum("...ij,dir,jr->...rd", 2 * dm, self.grad_basis, self.basis)

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
            return [prefix + "el_mat"]
        elif methodname == "ao_orb2dm":
            return []
        elif methodname == "get_vext":
            return [prefix + "basis_dvolume", prefix + "basis"]
        elif methodname == "get_grad_vext":
            return [prefix + "basis_dvolume", prefix + "grad_basis"]
        elif methodname == "get_lapl_vext":
            return [prefix + "basis_dvolume", prefix + "lapl_basis"]
        elif methodname == "get_vxc":
            assert self.xc is not None
            params = self.getparamnames("_dm2densinfo", prefix=prefix) + \
                self.getparamnames("get_vext", prefix=prefix) + \
                self.xc.getparamnames("get_vxc", prefix=prefix + "xc.")
            if self.xcfamily >= 2:
                params += self.getparamnames("get_grad_vext", prefix=prefix)
            if self.xcfamily >= 3:
                params += self.getparamnames("get_lapl_vext", prefix=prefix)
            return params
        elif methodname == "_dm2densinfo":
            params = [prefix + "basis"]
            if self.xcfamily >= 2:
                params += [prefix + "grad_basis"]
            if self.xcfamily >= 3:
                params += [prefix + "lapl_basis"]
            return params
        else:
            raise KeyError("getparamnames has no %s method" % methodname)
        # TODO: complete this
