from typing import List, Optional, Union, overload, Tuple
import torch
import xitorch as xt
import dqc.hamilton.intor as intor
from dqc.df.base_df import BaseDF
from dqc.df.dfmol import DFMol
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.utils.datastruct import AtomCGTOBasis, ValGrad, SpinParam, DensityFitInfo
from dqc.grid.base_grid import BaseGrid
from dqc.xc.base_xc import BaseXC
from dqc.utils.cache import Cache

class HamiltonCGTO(BaseHamilton):
    def __init__(self, atombases: List[AtomCGTOBasis], spherical: bool = True,
                 df: Optional[DensityFitInfo] = None,
                 efield: Optional[Tuple[torch.Tensor, ...]] = None,
                 cache: Optional[Cache] = None) -> None:
        self.atombases = atombases
        self.spherical = spherical
        self.libcint_wrapper = intor.LibcintWrapper(atombases, spherical)
        self.dtype = self.libcint_wrapper.dtype
        self.device = self.libcint_wrapper.device
        self._dfoptions = df
        if df is None:
            self._df: Optional[DFMol] = None
        else:
            self._df = DFMol(df, wrapper=self.libcint_wrapper)

        self._efield = efield
        self.is_grid_set = False
        self.is_ao_set = False
        self.is_grad_ao_set = False
        self.is_lapl_ao_set = False
        self.xc: Optional[BaseXC] = None
        self.xcfamily = 1
        self.is_built = False

        # initialize cache
        self._cache = cache if cache is not None else Cache.get_dummy()
        self._cache.add_cacheable_params(["overlap", "kinetic", "nuclattr", "efield0"])
        if self._df is None:
            self._cache.add_cacheable_params(["elrep"])

    @property
    def nao(self) -> int:
        return self.libcint_wrapper.nao()

    @property
    def kpts(self) -> torch.Tensor:
        raise TypeError("Isolated molecule Hamiltonian does not have kpts property")

    @property
    def df(self) -> Optional[BaseDF]:
        return self._df

    def build(self) -> BaseHamilton:
        # get the matrices (all (nao, nao), except el_mat)
        # these matrices have already been normalized
        with self._cache.open():

            # check the signature
            self._cache.check_signature({
                "atombases": self.atombases,
                "spherical": self.spherical,
                "dfoptions": self._dfoptions,
            })

            self.olp_mat = self._cache.cache("overlap", lambda: intor.overlap(self.libcint_wrapper))
            kin_mat = self._cache.cache("kinetic", lambda: intor.kinetic(self.libcint_wrapper))
            nucl_mat = self._cache.cache("nuclattr", lambda: intor.nuclattr(self.libcint_wrapper))
            self.nucl_mat = nucl_mat
            self.kinnucl_mat = kin_mat + nucl_mat

            # electric field integral
            if self._efield is not None:
                # (ndim, nao, nao)
                fac: float = 1.0
                for i in range(len(self._efield)):
                    fac *= i + 1
                    intor_fcn = lambda: intor.int1e("r0" * (i + 1), self.libcint_wrapper)
                    efield_mat_f = self._cache.cache(f"efield{i}", intor_fcn)
                    efield_mat = torch.einsum("dab,d->ab", efield_mat_f, self._efield[i])
                    self.kinnucl_mat = self.kinnucl_mat + efield_mat / fac

            if self._df is None:
                self.el_mat = self._cache.cache("elrep", lambda: intor.elrep(self.libcint_wrapper))  # (nao^4)
            else:
                self._df.build()
            self.is_built = True

        return self

    def get_nuclattr(self) -> xt.LinearOperator:
        # nucl_mat: (nao, nao)
        # return: (nao, nao)
        return xt.LinearOperator.m(self.nucl_mat, is_hermitian=True)

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
        if self._df is None:
            mat = torch.einsum("...ij,ijkl->...kl", dm, self.el_mat)
            mat = (mat + mat.transpose(-2, -1)) * 0.5  # reduce numerical instability
            return xt.LinearOperator.m(mat, is_hermitian=True)
        else:
            elrep = self._df.get_elrep(dm)
            return elrep

    @overload
    def get_exchange(self, dm: torch.Tensor) -> xt.LinearOperator:
        ...

    @overload
    def get_exchange(self, dm: SpinParam[torch.Tensor]) -> SpinParam[xt.LinearOperator]:
        ...

    def get_exchange(self, dm):
        # get the exchange operator
        # dm: (*BD, nao, nao)
        # el_mat: (nao, nao, nao, nao)
        # return: (*BD, nao, nao)
        if self._df is not None:
            raise RuntimeError("Exact exchange cannot be computed with density fitting")
        elif isinstance(dm, torch.Tensor):
            mat = -0.5 * torch.einsum("...jk,ijkl->...il", dm, self.el_mat)
            mat = (mat + mat.transpose(-2, -1)) * 0.5  # reduce numerical instability
            return xt.LinearOperator.m(mat, is_hermitian=True)
        else:  # dm is SpinParam
            # using the spin-scaling property of exchange energy
            return SpinParam(u=self.get_exchange(2 * dm.u),
                             d=self.get_exchange(2 * dm.d))

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
        self.dvolume = self.grid.get_dvolume()
        self.basis_dvolume = self.basis * self.dvolume  # (nao, ngrid)

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

    def get_lapl_kin_vext(self, lapl_vext: torch.Tensor, kin_vext: torch.Tensor) -> xt.LinearOperator:
        # get the linear operator for the laplacian and kinetic parts of the potential,
        # lapl_vext is the derivative of energy w.r.t. laplacian of the density
        # kin_vext is the derivative of energy w.r.t. kinetic energy density
        #     (i.e. 0.5 * sum((nabla phi)^2))
        # lapl_vext: (*BR, ngrid)
        # kin_vext: (*BR, ngrid)
        # return: (*BR, nao, nao)
        if not self.is_lapl_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, xc)` to call this function")
        # the equation below is obtained by calculating dExc/dD_ij where D_ij is
        # the density matrix
        # the equation for kinetic derivative is from eq. (26) https://doi.org/10.1063/1.4811270
        # but there is a missing factor half in it, see eq. (26) https://doi.org/10.1063/1.4967960

        lapl_dvol = lapl_vext * self.dvolume
        lapl_kin_dvol = (2 * lapl_vext + 0.5 * kin_vext) * self.dvolume
        mat1 = torch.einsum("...r,br,cr->...bc", lapl_dvol, self.basis, self.lapl_basis)
        mat1 = mat1 + mat1.transpose(-2, -1)  # + c.c.
        mat2 = torch.einsum("...r,dbr,dcr->...bc", lapl_kin_dvol, self.grad_basis, self.grad_basis)
        mat = mat1 + mat2
        return xt.LinearOperator.m(mat, is_hermitian=True)

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
            if self.xcfamily >= 4:  # MGGA
                assert potinfo.lapl is not None
                assert potinfo.kin is not None
                vxc_linop = vxc_linop + self.get_lapl_kin_vext(potinfo.lapl, potinfo.kin)

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
            if self.xcfamily >= 4:  # MGGA
                assert potinfo.u.lapl is not None
                assert potinfo.d.lapl is not None
                assert potinfo.u.kin is not None
                assert potinfo.d.kin is not None
                vxc_linop_u = vxc_linop_u + self.get_lapl_kin_vext(potinfo.u.lapl, potinfo.u.kin)
                vxc_linop_d = vxc_linop_d + self.get_lapl_kin_vext(potinfo.d.lapl, potinfo.d.kin)

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
            dens = self._get_dens_at_grid(dm)

            # calculate the density gradient
            gdens = None
            if family >= 2:  # requires gradient
                # (*BD, ngrid, ndim)
                # dm is multiplied by 2 because n(r) = sum (D_ij * phi_i * phi_j), thus
                # d.n(r) = sum (D_ij * d.phi_i * phi_j + D_ij * phi_i * d.phi_j)
                gdens = self._get_grad_dens_at_grid(dm)

            glapl: Optional[torch.Tensor] = None
            gkin: Optional[torch.Tensor] = None
            if family >= 4:
                glapl, gkin = self._get_lapl_kin_dens_at_grid(dm)

            # dens: (*BD, ngrid)
            # gdens: (*BD, ngrid, ndim)
            res = ValGrad(value=dens, grad=gdens, lapl=glapl, kin=gkin)
            return res

    def _get_dens_at_grid(self, dm: torch.Tensor) -> torch.Tensor:
        # get the density at the grid
        return torch.einsum("...ij,ir,jr->...r", dm, self.basis, self.basis)

    def _get_grad_dens_at_grid(self, dm: torch.Tensor) -> torch.Tensor:
        # get the gradient of density at the grid
        if not self.is_grad_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, gradlevel>=1)` to calculate the density gradient")
        # dm is multiplied by 2 because n(r) = sum (D_ij * phi_i * phi_j), thus
        # d.n(r) = sum (D_ij * d.phi_i * phi_j + D_ij * phi_i * d.phi_j)
        return torch.einsum("...ij,dir,jr->...rd", 2 * dm, self.grad_basis, self.basis)

    def _get_lapl_kin_dens_at_grid(self, dm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # calculate the laplacian of the density and kinetic energy density at the grid
        if not self.is_lapl_ao_set:
            raise RuntimeError("Please call `setup_grid(grid, gradlevel>=1)` to calculate the density gradient")
        dmt = dm.transpose(-2, -1)
        lapl_basis = torch.einsum("...ij,ir,jr->...r", (dm + dmt), self.lapl_basis, self.basis)
        grad_grad = torch.einsum("...ij,dir,djr->...r", dm, self.grad_basis, self.grad_basis)
        lapl = lapl_basis + 2 * grad_grad
        kin = grad_grad * 0.5
        return lapl, kin

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_kinnucl":
            return [prefix + "kinnucl_mat"]
        elif methodname == "get_nuclattr":
            return [prefix + "nucl_mat"]
        elif methodname == "get_overlap":
            return [prefix + "olp_mat"]
        elif methodname == "get_elrep":
            if self._df is None:
                return [prefix + "el_mat"]
            else:
                return self._df.getparamnames("get_elrep", prefix=prefix + "_df.")
        elif methodname == "get_exchange":
            return [prefix + "el_mat"]
        elif methodname == "ao_orb2dm":
            return []
        elif methodname == "get_vext":
            return [prefix + "basis_dvolume", prefix + "basis"]
        elif methodname == "get_grad_vext":
            return [prefix + "basis_dvolume", prefix + "grad_basis"]
        elif methodname == "get_lapl_kin_vext":
            return [prefix + "dvolume", prefix + "basis", prefix + "grad_basis",
                    prefix + "lapl_basis"]
        elif methodname == "get_vxc":
            assert self.xc is not None
            params = self.getparamnames("_dm2densinfo", prefix=prefix) + \
                self.getparamnames("get_vext", prefix=prefix) + \
                self.xc.getparamnames("get_vxc", prefix=prefix + "xc.")
            if self.xcfamily >= 2:
                params += self.getparamnames("get_grad_vext", prefix=prefix)
            if self.xcfamily >= 4:
                params += self.getparamnames("get_lapl_kin_vext", prefix=prefix)
            return params
        elif methodname == "_dm2densinfo":
            params = self.getparamnames("_get_dens_at_grid", prefix=prefix)
            if self.xcfamily >= 2:
                params += self.getparamnames("_get_grad_dens_at_grid", prefix=prefix)
            if self.xcfamily >= 4:
                params += self.getparamnames("_get_lapl_kin_dens_at_grid", prefix=prefix)
            return params
        elif methodname == "_get_dens_at_grid":
            return [prefix + "basis"]
        elif methodname == "_get_grad_dens_at_grid":
            return [prefix + "basis", prefix + "grad_basis"]
        elif methodname == "_get_lapl_kin_dens_at_grid":
            return [prefix + "basis", prefix + "grad_basis", prefix + "lapl_basis"]
        else:
            raise KeyError("getparamnames has no %s method" % methodname)
        # TODO: complete this
