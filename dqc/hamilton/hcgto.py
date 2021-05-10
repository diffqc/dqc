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
            # the einsum form below is to hack PyTorch's bug #57121
            # mat = -0.5 * torch.einsum("...jk,ijkl->...il", dm, self.el_mat)  # slower
            mat = -0.5 * torch.einsum("...il,ijkl->...ijk", dm, self.el_mat).sum(dim=-3)  # faster

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

        densinfo = SpinParam.apply_fcn(
            lambda dm_: self._dm2densinfo(dm_), dm)  # value: (*BD, nr)
        potinfo = self.xc.get_vxc(densinfo)  # value: (*BD, nr)
        vxc_linop = SpinParam.apply_fcn(
            lambda potinfo_: self._get_vxc_from_potinfo(potinfo_), potinfo)
        return vxc_linop

    def get_exc(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        assert self.xc is not None, "Please call .setup_grid with the xc object"

        # obtain the energy density per unit volume
        densinfo = SpinParam.apply_fcn(
            lambda dm_: self._dm2densinfo(dm_), dm)  # (spin) value: (*BD, nr)
        edens = self.xc.get_edensityxc(densinfo)  # (*BD, nr)

        return torch.sum(self.grid.get_dvolume() * edens, dim=-1)

    def _dm2densinfo(self, dm: torch.Tensor) -> ValGrad:
        # dm: (*BD, nao, nao), Hermitian
        # family: 1 for LDA, 2 for GGA, 3 for MGGA
        # self.basis: (nao, ngrid)

        # dm @ ao will be used in every case
        dmdmt = (dm + dm.transpose(-2, -1)) * 0.5  # (*BD, nao, nao)
        dmao = torch.matmul(dmdmt, self.basis)  # (*BD, nao, nr)

        # calculate the density
        dens = torch.einsum("...ir,ir->...r", dmao, self.basis)

        # calculate the density gradient
        gdens: Optional[torch.Tensor] = None
        if self.xcfamily == 2 or self.xcfamily == 4:  # GGA or MGGA
            if not self.is_grad_ao_set:
                msg = "Please call `setup_grid(grid, gradlevel>=1)` to calculate the density gradient"
                raise RuntimeError(msg)

            gdens = torch.zeros((*dm.shape[:-2], 3, self.basis.shape[-1]),
                                dtype=self.dtype, device=self.device)  # (..., ndim, ngrid)
            gdens[..., 0, :] = torch.einsum("...ir,ir->...r", dmao, self.grad_basis[0]) * 2
            gdens[..., 1, :] = torch.einsum("...ir,ir->...r", dmao, self.grad_basis[1]) * 2
            gdens[..., 2, :] = torch.einsum("...ir,ir->...r", dmao, self.grad_basis[2]) * 2

        lapldens: Optional[torch.Tensor] = None
        kindens: Optional[torch.Tensor] = None
        if self.xcfamily == 4:
            # calculate the laplacian of the density and kinetic energy density at the grid
            if not self.is_lapl_ao_set:
                msg = "Please call `setup_grid(grid, gradlevel>=2)` to calculate the density gradient"
                raise RuntimeError(msg)
            lapl_basis = torch.einsum("...ir,ir->...r", dmao, self.lapl_basis)
            grad_grad = torch.einsum("...ir,ir->...r", torch.matmul(dmdmt, self.grad_basis[0]), self.grad_basis[0])
            grad_grad += torch.einsum("...ir,ir->...r", torch.matmul(dmdmt, self.grad_basis[1]), self.grad_basis[1])
            grad_grad += torch.einsum("...ir,ir->...r", torch.matmul(dmdmt, self.grad_basis[2]), self.grad_basis[2])
            # pytorch's "...ij,ir,jr->...r" is really slow for large matrix
            # grad_grad = torch.einsum("...ij,ir,jr->...r", dmdmt, self.grad_basis[0], self.grad_basis[0])
            # grad_grad += torch.einsum("...ij,ir,jr->...r", dmdmt, self.grad_basis[1], self.grad_basis[1])
            # grad_grad += torch.einsum("...ij,ir,jr->...r", dmdmt, self.grad_basis[2], self.grad_basis[2])
            lapldens = (lapl_basis + grad_grad) * 2
            kindens = grad_grad * 0.5

        # dens: (*BD, ngrid)
        # gdens: (*BD, ndim, ngrid)
        res = ValGrad(value=dens, grad=gdens, lapl=lapldens, kin=kindens)
        return res

    def _get_vxc_from_potinfo(self, potinfo: ValGrad) -> xt.LinearOperator:
        # obtain the vxc operator from the potential information

        vb = potinfo.value.unsqueeze(-2) * self.basis
        if self.xcfamily in [2, 4]:  # GGA or MGGA
            assert potinfo.grad is not None  # (..., ndim, nrgrid)
            vgrad = potinfo.grad * 2
            vb += torch.einsum("...r,ar->...ar", vgrad[..., 0, :], self.grad_basis[0])
            vb += torch.einsum("...r,ar->...ar", vgrad[..., 1, :], self.grad_basis[1])
            vb += torch.einsum("...r,ar->...ar", vgrad[..., 2, :], self.grad_basis[2])
        if self.xcfamily == 4:  # MGGA
            assert potinfo.lapl is not None  # (..., nrgrid)
            assert potinfo.kin is not None
            vb += 2 * potinfo.lapl.unsqueeze(-2) * self.lapl_basis

        # calculating the matrix from multiplication with the basis
        mat = torch.matmul(vb, self.basis_dvolume.transpose(-2, -1))

        if self.xcfamily == 4:  # MGGA
            assert potinfo.lapl is not None  # (..., nrgrid)
            assert potinfo.kin is not None
            lapl_kin_dvol = (2 * potinfo.lapl + 0.5 * potinfo.kin) * self.dvolume
            mat += torch.einsum("...r,br,cr->...bc", lapl_kin_dvol, self.grad_basis[0], self.grad_basis[0])
            mat += torch.einsum("...r,br,cr->...bc", lapl_kin_dvol, self.grad_basis[1], self.grad_basis[1])
            mat += torch.einsum("...r,br,cr->...bc", lapl_kin_dvol, self.grad_basis[2], self.grad_basis[2])

        # construct the Hermitian linear operator
        mat = (mat + mat.transpose(-2, -1)) * 0.5
        vxc_linop = xt.LinearOperator.m(mat, is_hermitian=True)
        return vxc_linop

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
            return self.getparamnames("_dm2densinfo", prefix=prefix) + \
                self.getparamnames("_get_vxc_from_potinfo", prefix=prefix) + \
                self.xc.getparamnames("get_vxc", prefix=prefix + "xc.")
        elif methodname == "_dm2densinfo":
            params = [prefix + "basis"]
            if self.xcfamily == 2 or self.xcfamily == 4:
                params += [prefix + "grad_basis"]
            if self.xcfamily == 4:
                params += [prefix + "lapl_basis"]
            return params
        elif methodname == "_get_vxc_from_potinfo":
            params = [prefix + "basis", prefix + "basis_dvolume"]
            if self.xcfamily in [2, 4]:
                params += [prefix + "grad_basis"]
            if self.xcfamily == 4:
                params += [prefix + "lapl_basis", prefix + "dvolume"]
            return params
        else:
            raise KeyError("getparamnames has no %s method" % methodname)
        # TODO: complete this
