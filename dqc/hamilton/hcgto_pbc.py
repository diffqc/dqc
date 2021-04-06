from typing import List, Optional, Union, overload, Dict
import torch
import numpy as np
import xitorch as xt
import dqc.hamilton.intor as intor
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.df.base_df import BaseDF
from dqc.df.dfpbc import DFPBC
from dqc.utils.datastruct import CGTOBasis, AtomCGTOBasis, SpinParam, DensityFitInfo
from dqc.utils.pbc import unweighted_coul_ft, get_gcut
from dqc.grid.base_grid import BaseGrid
from dqc.xc.base_xc import BaseXC
from dqc.hamilton.intor.lattice import Lattice

class HamiltonCGTO_PBC(BaseHamilton):
    """
    Hamiltonian with contracted Gaussian type orbitals in a periodic boundary
    condition systems.
    The calculation of Hamiltonian components follow the reference:
    Sun, et al., J. Chem. Phys. 147, 164119 (2017)
    https://doi.org/10.1063/1.4998644
    """
    def __init__(self, atombases: List[AtomCGTOBasis],
                 latt: Lattice,
                 kpts: Optional[torch.Tensor] = None,
                 wkpts: Optional[torch.Tensor] = None,  # weights of k-points to get the density
                 spherical: bool = True,
                 df: Optional[DensityFitInfo] = None,
                 lattsum_opt: Optional[Union[intor.PBCIntOption, Dict]] = None) -> None:
        self._atombases = atombases
        self._spherical = spherical
        self._lattice = latt
        # alpha for the compensating charge
        # TODO: calculate eta properly or put it in lattsum_opt
        self._eta = 0.2
        self._eta = 0.46213127322256375  # temporary to follow pyscf.df
        # lattice sum integral options
        if lattsum_opt is None:
            self._lattsum_opt = intor.PBCIntOption()
        elif isinstance(lattsum_opt, dict):
            self._lattsum_opt = intor.PBCIntOption(**lattsum_opt)
        else:
            self._lattsum_opt = lattsum_opt

        self._basiswrapper = intor.LibcintWrapper(
            atombases, spherical=spherical, lattice=latt)
        self.dtype = self._basiswrapper.dtype
        self.device = self._basiswrapper.device

        # set the default k-points and their weights
        self._kpts = kpts if kpts is not None else \
            torch.zeros((1, 3), dtype=self.dtype, device=self.device)
        nkpts = self._kpts.shape[0]
        # default weights are just 1/nkpts (nkpts,)
        self._wkpts = wkpts if wkpts is not None else \
            torch.ones((nkpts,), dtype=self.dtype, device=self.device) / nkpts

        assert self._wkpts.shape[0] == self._kpts.shape[0]
        assert self._wkpts.ndim == 1
        assert self._kpts.ndim == 2

        if df is None:
            self._df: Optional[BaseDF] = None
        else:
            self._df = DFPBC(dfinfo=df, wrapper=self._basiswrapper, kpts=self._kpts,
                             wkpts=self._wkpts, eta=self._eta,
                             lattsum_opt=self._lattsum_opt)

        self._is_built = False

    @property
    def nao(self) -> int:
        return self._basiswrapper.nao()

    @property
    def kpts(self) -> torch.Tensor:
        return self._kpts

    @property
    def df(self) -> Optional[BaseDF]:
        return self._df

    def build(self) -> BaseHamilton:
        if self._df is None:
            raise NotImplementedError(
                "Periodic boundary condition without density fitting is not implemented")
        assert isinstance(self._df, BaseDF)
        # (nkpts, nao, nao)
        self._olp_mat = intor.pbc_overlap(self._basiswrapper, kpts=self._kpts,
                                          options=self._lattsum_opt)
        self._kin_mat = intor.pbc_kinetic(self._basiswrapper, kpts=self._kpts,
                                          options=self._lattsum_opt)
        self._nucl_mat = self._calc_nucl_attr()
        self._kinnucl_mat = self._kin_mat + self._nucl_mat
        self._df.build()
        self._is_built = True
        return self

    def get_nuclattr(self) -> xt.LinearOperator:
        # return: (nkpts, nao, nao)
        return xt.LinearOperator.m(self._nucl_mat, is_hermitian=True)

    def get_kinnucl(self) -> xt.LinearOperator:
        # kinnucl_mat: (nkpts, nao, nao)
        # return: (nkpts, nao, nao)
        return xt.LinearOperator.m(self._kinnucl_mat, is_hermitian=True)

    def get_overlap(self) -> xt.LinearOperator:
        # olp_mat: (nkpts, nao, nao)
        # return: (nkpts, nao, nao)
        return xt.LinearOperator.m(self._olp_mat, is_hermitian=True)

    def get_elrep(self, dm: torch.Tensor) -> xt.LinearOperator:
        # dm: (nkpts, nao, nao)
        # return: (nkpts, nao, nao)
        assert self._df is not None
        return self._df.get_elrep(dm)

    def ao_orb2dm(self, orb: torch.Tensor, orb_weight: torch.Tensor) -> torch.Tensor:
        # convert the atomic orbital to the density matrix

        # orb: (nkpts, nao, norb)
        # orb_weight: (norb)
        # return: (nkpts, nao, nao)
        dtype = orb.dtype
        res = torch.einsum("kao,o,kbo->kab", orb, orb_weight.to(dtype), orb.conj())
        return res

    def aodm2dens(self, dm: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (*BR, ndim)
        # dm: (*BD, nkpts, nao, nao)
        # returns: (*BRD)

        nao = dm.shape[-1]
        nkpts = self._kpts.shape[0]
        xyzshape = xyz.shape  # (*BR, ndim)

        # basis: (nkpts, nao, *BR)
        xyz1 = xyz.reshape(-1, xyzshape[-1])  # (BR=ngrid, ndim)
        # ao1: (nkpts, nao, ngrid)
        ao1 = intor.pbc_eval_gto(self._basiswrapper, xyz1, kpts=self._kpts, options=self._lattsum_opt)
        ao1 = torch.movedim(ao1, -1, 0).reshape(*xyzshape[:-1], nkpts, nao)  # (*BR, nkpts, nao)

        # dens = torch.einsum("...ka,...kb,...kab,k->...", ao1, ao1.conj(), dm, self._wkpts)
        densk = torch.matmul(dm, ao1.conj().unsqueeze(-1))  # (*BRD, nkpts, nao, 1)
        densk = torch.matmul(ao1.unsqueeze(-2), densk).squeeze(-1).squeeze(-1)  # (*BRD, nkpts)
        assert densk.imag.abs().max() < 1e-9, "The density should be real at this point"

        dens = torch.einsum("...k,k->...", densk.real, self._wkpts)  # (*BRD)
        return dens

    ############### grid-related ###############
    def setup_grid(self, grid: BaseGrid, xc: Optional[BaseXC] = None) -> None:
        # save the family and save the xc
        pass

    def get_vext(self, vext: torch.Tensor) -> xt.LinearOperator:
        # vext: (*BR, ngrid)
        pass

    def get_grad_vext(self, grad_vext: torch.Tensor) -> xt.LinearOperator:
        # grad_vext: (*BR, ngrid, ndim)
        pass

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
        pass

    def get_exc(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        pass

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if False:
            pass
        else:
            raise KeyError("getparamnames has no %s method" % methodname)

    ################ private methods ################
    def _calc_nucl_attr(self) -> torch.Tensor:
        # calculate the nuclear attraction matrix
        # this follows the equation (31) in Sun, et al., J. Chem. Phys. 147 (2017)

        # construct the fake nuclei atombases for nuclei
        # (in this case, we assume each nucleus is a very sharp s-type orbital)
        nucl_atbases = self._create_fake_nucl_bases(alpha=1e16, chargemult=1)
        # add a compensating charge
        cnucl_atbases = self._create_fake_nucl_bases(alpha=self._eta, chargemult=-1)
        # real charge + compensating charge
        nucl_atbases_all = nucl_atbases + cnucl_atbases
        nucl_wrapper = intor.LibcintWrapper(
            nucl_atbases_all, spherical=self._spherical, lattice=self._lattice)
        cnucl_wrapper = intor.LibcintWrapper(
            cnucl_atbases, spherical=self._spherical, lattice=self._lattice)
        natoms = nucl_wrapper.nao() // 2

        # construct the k-points ij
        # duplicating kpts to have shape of (nkpts, 2, ndim)
        kpts_ij = self._kpts.unsqueeze(-2) * torch.ones((2, 1), dtype=self.dtype, device=self.device)

        ############# 1st part of nuclear attraction: short range #############
        # get the 1st part of the nuclear attraction: the charge and compensating charge
        # nuc1: (nkpts, nao, nao, 2 * natoms)
        # nuc1 is not hermitian
        basiswrapper1, nucl_wrapper1 = intor.LibcintWrapper.concatenate(self._basiswrapper, nucl_wrapper)
        nuc1_c = intor.pbc_coul3c(basiswrapper1, other=basiswrapper1,
                                  auxwrapper=nucl_wrapper1, kpts_ij=kpts_ij,
                                  options=self._lattsum_opt)
        nuc1 = -nuc1_c[..., :natoms] + nuc1_c[..., natoms:]
        nuc1 = torch.sum(nuc1, dim=-1)  # (nkpts, nao, nao)

        # add vbar for 3 dimensional cell
        # vbar is the interaction between the background charge and the
        # compensating function.
        # https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/pbc/df/aft.py#L239
        nucbar = sum([-atb.atomz / self._eta for atb in self._atombases])
        nuc1_b = -nucbar * np.pi / self._lattice.volume() * self._olp_mat
        nuc1 = nuc1 + nuc1_b

        ############# 2nd part of nuclear attraction: long range #############
        # get the 2nd part from the Fourier Transform
        # get the G-points, choosing min because the two FTs are multiplied
        gcut = get_gcut(self._lattsum_opt.precision,
                        wrappers=[cnucl_wrapper, self._basiswrapper],
                        reduce="min")
        # gvgrids: (ngv, ndim), gvweights: (ngv,)
        gvgrids, gvweights = self._lattice.get_gvgrids(gcut)

        # the compensating charge's Fourier Transform
        # TODO: split gvgrids and gvweights to reduce the memory usage
        cnucl_ft = intor.eval_gto_ft(cnucl_wrapper, gvgrids)  # (natoms, ngv)
        # overlap integral of the electron basis' Fourier Transform
        cbas_ft = intor.pbcft_overlap(
            self._basiswrapper, Gvgrid=-gvgrids, kpts=self._kpts,
            options=self._lattsum_opt)  # (nkpts, nao, nao, ngv)
        # coulomb kernel Fourier Transform
        coul_ft = unweighted_coul_ft(gvgrids) * gvweights  # (ngv,)
        coul_ft = coul_ft.to(cbas_ft.dtype)  # cast to complex

        # optimized by opt_einsum
        # nuc2 = -torch.einsum("tg,kabg,g->kab", cnucl_ft, cbas_ft, coul_ft)
        nuc2_temp = torch.einsum("g,tg->g", coul_ft, cnucl_ft)
        nuc2 = -torch.einsum("g,kabg->kab", nuc2_temp, cbas_ft)  # (nkpts, nao, nao)
        # print((nuc2 - nuc2.conj().transpose(-2, -1)).abs().max())  # check hermitian-ness

        # get the total contribution from the short range and long range
        nuc = nuc1 + nuc2

        # symmetrize for more stable numerical calculation
        nuc = (nuc + nuc.conj().transpose(-2, -1)) * 0.5
        return nuc

    def _create_fake_nucl_bases(self, alpha: float, chargemult: int) -> List[AtomCGTOBasis]:
        # create a list of basis (of s-type) at every nuclei positions
        res: List[AtomCGTOBasis] = []
        alphas = torch.tensor([alpha], dtype=self.dtype, device=self.device)
        # normalizing so the integral of the cgto is 1
        # 0.5 / np.sqrt(np.pi) * 2 / scipy.special.gamma(1.5) * alphas ** 1.5
        norm_coeff = 0.6366197723675814 * alphas ** 1.5
        for atb in self._atombases:
            # put the charge in the coefficients
            coeffs = atb.atomz * norm_coeff
            basis = CGTOBasis(angmom=0, alphas=alphas, coeffs=coeffs, normalized=True)
            res.append(AtomCGTOBasis(atomz=0, bases=[basis], pos=atb.pos))
        return res
