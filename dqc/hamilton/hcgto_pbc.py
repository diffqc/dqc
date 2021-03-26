from typing import List, Optional, Union, overload
import torch
import numpy as np
import xitorch as xt
import dqc.hamilton.intor as intor
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.hamilton.intor.utils import estimate_ke_cutoff
from dqc.utils.datastruct import CGTOBasis, AtomCGTOBasis, SpinParam, DensityFitInfo
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
                 spherical: bool = True,
                 df: Optional[DensityFitInfo] = None,
                 lattsum_opt: Optional[intor.PBCIntOption] = None) -> None:
        self._atombases = atombases
        self._spherical = spherical
        self._lattice = latt
        self._df = df
        self._eta = 0.2  # alpha for the compensating charge
        # lattice sum integral options
        self._lattsum_opt = intor.PBCIntOption() if lattsum_opt is None else lattsum_opt

        self._basiswrapper = intor.LibcintWrapper(
            atombases, spherical=spherical, lattice=latt)
        self.dtype = self._basiswrapper.dtype
        self.device = self._basiswrapper.device
        self._kpts = kpts if kpts is not None else \
            torch.zeros((1, 3), dtype=self.dtype, device=self.device)

        self._is_built = False

    @property
    def nao(self) -> int:
        return self._basiswrapper.nao()

    def build(self) -> BaseHamilton:
        if self._df is None:
            raise NotImplementedError(
                "Periodic boundary condition without density fitting is not implemented")
        # (nkpts, nao, nao)
        self._olp_mat = intor.pbc_overlap(self._basiswrapper, kpts=self._kpts)
        self._kin_mat = intor.pbc_kinetic(self._basiswrapper, kpts=self._kpts)
        self._nucl_mat = self._calc_nucl_attr()
        pass  # nucl_attr and elrep
        self._is_built = True
        return self

    def get_kinnucl(self) -> xt.LinearOperator:
        # kinnucl_mat: (nao, nao)
        # return: (nao, nao)
        return xt.LinearOperator.m(self._kinnucl_mat, is_hermitian=True)  # ???

    def get_overlap(self) -> xt.LinearOperator:
        # olp_mat: (nao, nao)
        # return: (nao, nao)
        return xt.LinearOperator.m(self._olp_mat, is_hermitian=True)

    def get_elrep(self, dm: torch.Tensor) -> xt.LinearOperator:
        # dm: (*BD, nao, nao)
        # elrep_mat: (nao, nao, nao, nao)
        # return: (*BD, nao, nao)
        pass

    def ao_orb2dm(self, orb: torch.Tensor, orb_weight: torch.Tensor) -> torch.Tensor:
        # convert the atomic orbital to the density matrix
        # in CGTO, it is U.W.U^T

        # orb: (*BO, nao, norb)
        # orb_weight: (*BW, norb)
        # return: (*BOW, nao, nao)
        pass

    def aodm2dens(self, dm: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (*BR, ndim)
        # dm: (*BD, nao, nao)
        # returns: (*BRD)
        pass

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
        nucl_atbases = self._create_fake_nucl_bases(alpha=1e6, chargemult=1)
        # add a compensating charge
        cnucl_atbases = self._create_fake_nucl_bases(alpha=self._eta, chargemult=-1)
        # real charge + compensating charge
        nucl_atbases_all = nucl_atbases + cnucl_atbases
        nucl_wrapper = intor.LibcintWrapper(
            nucl_atbases_all, spherical=self._spherical, lattice=self._lattice)
        cnucl_wrapper = nucl_wrapper[len(nucl_wrapper) // 2:]
        natoms = nucl_wrapper.nao() // 2

        # construct the k-points ij
        # duplicating kpts to have shape of (nkpts, 2, ndim)
        kpts_ij = self._kpts.unsqueeze(-2) * torch.ones((2, 1), dtype=self.dtype, device=self.device)

        ############# 1st part of nuclear attraction: short range #############
        # get the 1st part of the nuclear attraction: the charge and compensating charge
        # nuc1: (nao, nao, 2 * natoms)
        basiswrapper1, nucl_wrapper1 = intor.LibcintWrapper.concatenate(self._basiswrapper, nucl_wrapper)
        nuc1_c = intor.pbc_coul3c(basiswrapper1, other=basiswrapper1,
                                  auxwrapper=nucl_wrapper1, kpts_ij=kpts_ij,
                                  options=self._lattsum_opt)
        nuc1 = -nuc1_c[..., :natoms] + nuc1_c[..., natoms:]

        ############# 2nd part of nuclear attraction: long range #############
        # get the 2nd part from the Fourier Transform
        # get the G-points
        coeffs_cnucl, alphas_cnucl, _ = cnucl_wrapper.params
        coeffs_basis, alphas_basis, _ = self._basiswrapper.params
        kecut_cnucl = estimate_ke_cutoff(self._lattsum_opt.precision, coeffs_cnucl, alphas_cnucl)
        kecut_basis = estimate_ke_cutoff(self._lattsum_opt.precision, coeffs_basis, alphas_basis)
        kecut = max(kecut_cnucl, kecut_basis)
        # gvgrids: (ngv, ndim), gvweights: (ngv,)
        gvgrids, gvweights = self._lattice.get_gvgrids(kecut, exclude_zeros=True)

        # the compensating charge's Fourier Transform
        # TODO: split gvgrids and gvweights to reduce the memory usage
        cnucl_ft = intor.eval_gto_ft(cnucl_wrapper, gvgrids)  # (natoms, ngv)
        # overlap integral of the electron basis' Fourier Transform
        cbas_ft = intor.pbcft_overlap(
            self._basiswrapper, Gvgrid=-gvgrids, kpts=self._kpts,
            options=self._lattsum_opt)  # (nkpts, nao, nao, ngv)
        # coulomb kernel Fourier Transform
        coul_ft = 4 * np.pi / (gvgrids * gvgrids) * gvweights  # (ngv)
        # nuc2: (nkpts, nao, nao)
        nuc2 = -torch.einsum("tg,kabg,g->kab", cnucl_ft, cbas_ft, coul_ft)

        # get the total contribution from the short range and long range
        nuc = nuc1 + nuc2
        return nuc

    def _create_fake_nucl_bases(self, alpha: float, chargemult: int) -> List[AtomCGTOBasis]:
        # create a list of basis (of s-type) at every nuclei positions
        res: List[AtomCGTOBasis] = []
        alphas = torch.tensor([alpha], dtype=self.dtype, device=self.device)
        # normalizing so the integral of the cgto is 1
        norm_coeff = 1.4366969770013325 * alphas ** 1.5
        for atb in self._atombases:
            # put the charge in the coefficients
            coeffs = atb.atomz * norm_coeff
            basis = CGTOBasis(angmom=0, alphas=alphas, coeffs=coeffs, normalized=True)
            res.append(AtomCGTOBasis(atomz=0, bases=[basis], pos=atb.pos))
        return res
