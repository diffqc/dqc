from typing import List, Optional, Union, overload, Tuple, Dict
import torch
import numpy as np
import xitorch as xt
import dqc.hamilton.intor as intor
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.hamilton.intor.utils import estimate_g_cutoff
from dqc.utils.datastruct import CGTOBasis, AtomCGTOBasis, SpinParam, DensityFitInfo
from dqc.utils.misc import gaussian_int
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
                 lattsum_opt: Optional[Union[intor.PBCIntOption, Dict]] = None) -> None:
        self._atombases = atombases
        self._spherical = spherical
        self._lattice = latt
        self._df = df
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
        self._kinnucl_mat = self._kin_mat + self._nucl_mat
        self._elrep_mat, self._elrep_mat_3c = self._calc_elrep_df(self._df)
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
        gcut = self._get_gcut(cnucl_wrapper, self._basiswrapper, reduce="min")
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
        coul_ft = self._unweighted_coul_ft(gvgrids) * gvweights  # (ngv,)
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

    def _calc_elrep_df(self, df: DensityFitInfo) -> Tuple[torch.Tensor, torch.Tensor]:
        # calculate the matrices required to calculate the electron repulsion operator
        # i.e. the 3-centre 2-electron integrals (short + long range) and j3c @ (j2c^-1)
        method = df.method.lower()
        df_auxbases = _renormalize_auxbases(df.auxbases)
        aux_comp_bases = self._create_compensating_bases(df_auxbases, eta=self._eta)
        fuse_aux_bases = df_auxbases + aux_comp_bases
        fuse_aux_wrapper = intor.LibcintWrapper(fuse_aux_bases, spherical=self._spherical,
                                                lattice=self._lattice)
        aux_comp_wrapper = intor.LibcintWrapper(aux_comp_bases, spherical=self._spherical,
                                                lattice=self._lattice)
        aux_wrapper = intor.LibcintWrapper(df_auxbases, spherical=self._spherical,
                                           lattice=self._lattice)
        print("auxwrapper atm bas env:")
        print(aux_wrapper.atm_bas_env)
        print("aux_comp_wrapper atm bas env:")
        print(aux_comp_wrapper.atm_bas_env)
        nxcao = aux_comp_wrapper.nao()  # number of aux compensating basis wrapper
        nxao = fuse_aux_wrapper.nao() - nxcao  # number of aux basis wrapper
        assert nxcao == nxao

        # only gaussian density fitting is implemented at the moment
        if method != "gdf":
            raise NotImplementedError("Density fitting that is not %s is not implemented" % df.method)

        # get the k-points needed for the integrations
        kpts_ij = _combine_kpts_to_kpts_ij(self._kpts)  # (nkpts_ij, 2, ndim)
        kpts_reduce = _reduce_kpts_ij(kpts_ij)  # (nkpts_ij, ndim)
        nkpts_ij = kpts_ij.shape[0]

        ######################## short-range integrals ########################
        ############# 3-centre 2-electron integral #############
        _basisw, _fusew = intor.LibcintWrapper.concatenate(self._basiswrapper, fuse_aux_wrapper)
        # (nkpts_ij, nao, nao, nxao+nxcao)
        j3c_short_f = intor.pbc_coul3c(_basisw, auxwrapper=_fusew, kpts_ij=kpts_ij)
        j3c_short = j3c_short_f[..., :nxao] - j3c_short_f[..., nxao:]  # (nkpts_ij, nao, nao, nxao)

        ############# 2-centre 2-electron integrals #############
        # (nkpts_unique, nxao+nxcao, nxao+nxcao)
        j2c_short_f = intor.pbc_coul2c(fuse_aux_wrapper, kpts=kpts_reduce)
        # j2c_short: (nkpts_unique, nxao, nxao)
        j2c_short = j2c_short_f[..., :nxao, :nxao] + j2c_short_f[..., nxao:, nxao:] \
                    - j2c_short_f[..., :nxao, nxao:] - j2c_short_f[..., nxao:, :nxao]

        ######################## long-range integrals ########################
        # only use the compensating wrapper as the gcut
        gcut = self._get_gcut(aux_comp_wrapper)
        # gvgrids: (ngv, ndim), gvweights: (ngv,)
        gvgrids, gvweights = self._lattice.get_gvgrids(gcut)
        ngv = gvgrids.shape[0]
        gvk = gvgrids.unsqueeze(-2) + kpts_reduce  # (ngv, nkpts_ij, ndim)
        gvk = gvk.view(-1, gvk.shape[-1])  # (ngv * nkpts_ij, ndim)

        # get the fourier transform variables
        # TODO: iterate over ngv axis
        # ft of the compensating basis
        comp_ft = intor.eval_gto_ft(aux_comp_wrapper, gvk)  # (nxcao, ngv * nkpts_ij)
        comp_ft = comp_ft.view(-1, ngv, nkpts_ij)  # (nxcao, ngv, nkpts_ij)
        # ft of the auxiliary basis
        auxb_ft_c = intor.eval_gto_ft(aux_wrapper, gvk)  # (nxao, ngv * nkpts_ij)
        auxb_ft_c = auxb_ft_c.view(-1, ngv, nkpts_ij)  # (nxao, ngv, nkpts_ij)
        auxb_ft = auxb_ft_c - comp_ft  # (nxao, ngv, nkpts_ij)
        # # ft of the overlap integral of the basis
        # aoao_ft = intor.pbcft_overlap(
        #     self._basiswrapper, Gvgrid=gvk, kpts=self._kpts,
        #     options=self._lattsum_opt)  # (nkpts, nao, nao, ngv * nkpts_ij)
        # ft of the coulomb kernel
        coul_ft = self._unweighted_coul_ft(gvk)  # (ngv * nkpts_ij,)
        coul_ft = coul_ft.to(comp_ft.dtype).view(ngv, nkpts_ij) * gvweights.unsqueeze(-1)  # (ngv, nkpts_ij)

        # 1: (nkpts_ij, nxao, nxao)
        pattern = "gi,xgi,ygi->ixy"
        j2c_long  = torch.einsum(pattern, coul_ft, comp_ft.conj(), auxb_ft)
        # 2: (nkpts_ij, nxao, nxao)
        j2c_long += torch.einsum(pattern, coul_ft, auxb_ft.conj(), comp_ft)
        # 3: (nkpts_ij, nxao, nxao)
        j2c_long += torch.einsum(pattern, coul_ft, comp_ft.conj(), comp_ft)

        # TODO: complete this
        j3c_long = 0

        ######################## combining integrals ########################
        j2c = j2c_short + j2c_long  # (nkpts_ij, nxao, nxao)
        j3c = j3c_short + j3c_long  # (nkpts_ij, nao, nao, nxao)
        el_mat = torch.matmul(j3c, torch.inverse(j2c.unsqueeze(1)))  # (nkpts_ij, nao, nao, nxao)
        return el_mat, j3c

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

    def _create_compensating_bases(self, atombases: List[AtomCGTOBasis], eta: float) -> List[AtomCGTOBasis]:
        # create the list of atom bases containing the compensating basis with
        # given `eta` as the exponentials
        # see make_modchg_basis in
        # https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/pbc/df/df.py#L116

        # pre-calculate the norms up to angmom 6
        half_sph_norm = 0.5 / np.sqrt(np.pi)
        norms = [half_sph_norm / gaussian_int(2 * angmom + 2, eta) for angmom in range(7)]
        norms_t = [torch.tensor([nrm], dtype=self.dtype, device=self.device) for nrm in norms]

        res: List[AtomCGTOBasis] = []
        alphas = torch.tensor([eta], dtype=self.dtype, device=self.device)
        for atb in atombases:
            # TODO: use reduced bases to optimize the integration time
            # angmoms = set(bas.angmom for bas in atb.bases)
            # bases = [
            #     CGTOBasis(angmom=angmom, alphas=alphas, coeffs=norms[angmom], normalized=True) \
            #     for angmom in angmoms
            # ]
            bases: List[CGTOBasis] = []
            for bas in atb.bases:
                # calculate the integral of the basis
                int1 = gaussian_int(bas.angmom * 2 + 2, bas.alphas)
                s = torch.sum(bas.coeffs * int1) / half_sph_norm

                # set the coefficients of the compensating basis to have
                # the same integral
                coeffs = s * norms_t[bas.angmom]
                b2 = CGTOBasis(angmom=bas.angmom, alphas=alphas,
                               coeffs=coeffs,
                               normalized=True)
                bases.append(b2)
            res.append(AtomCGTOBasis(atomz=0, bases=bases, pos=atb.pos))
        return res

    def _unweighted_coul_ft(self, gvgrids: torch.Tensor) -> torch.Tensor:
        # Returns the unweighted fourier transform of the coulomb kernel: 4*pi/|gv|^2
        # If |gv| == 0, then it is 0.
        # gvgrids: (ngv, ndim)
        # returns: (ngv,)
        gnorm2 = torch.einsum("xd,xd->x", gvgrids, gvgrids)
        gnorm2[gnorm2 < 1e-12] = float("inf")
        coulft = 4 * np.pi / gnorm2
        return coulft

    def _get_gcut(self, *wrappers: intor.LibcintWrapper, reduce="min") -> float:
        # get the G-point cut-off from the given wrappers where the FT
        # eval/integration is going to be performed
        gcuts: List[float] = []
        for wrapper in wrappers:
            # TODO: using params here can be confusing because wrapper.params
            # returns all parameters (even if it is a subset)
            coeffs, alphas, _ = wrapper.params
            gcut_wrap = estimate_g_cutoff(self._lattsum_opt.precision, coeffs, alphas)
            gcuts.append(gcut_wrap)
        if len(gcuts) == 1:
            return gcuts[0]
        if reduce == "min":
            return min(*gcuts)
        elif reduce == "max":
            return max(*gcuts)
        else:
            raise ValueError("Unknown reduce: %s" % reduce)

def _combine_kpts_to_kpts_ij(kpts: torch.Tensor) -> torch.Tensor:
    # combine the k-points into pair of k-points
    # kpts: (nkpts, ndim)
    # return: (nkpts_ij, 2, ndim) where nkpts_ij = nkpts ** 2
    nkpts, ndim = kpts.shape
    kpts_ij = torch.zeros((nkpts, nkpts, 2, ndim), dtype=kpts.dtype, device=kpts.device)
    kpts_ij[:, :, 0, :] = kpts.unsqueeze(1)
    kpts_ij[:, :, 1, :] = kpts.unsqueeze(0)
    kpts_ij = kpts_ij.view(-1, 2, ndim)
    return kpts_ij

def _reduce_kpts_ij(kpts_ij: torch.Tensor) -> torch.Tensor:
    # get the value of kpts_reduce = kpts_i - kpts_j.
    # however, as it might contain repeated k-points, only returns the unique
    # value of kpts_reduce and the inverse index that can be used to reconstruct
    # the original kpts_reduce
    # kpts_ij: (nkpts_ij, 2, ndim)
    # kpts_reduce: (nkpts_reduce, ndim)

    # TODO: optimize this by using unique!
    kpts_reduce = kpts_ij[..., 0, :] - kpts_ij[..., 1, :]  # (nkpts_ij, ndim)
    # inverse_idxs = torch.arange(kpts_reduce.shape[0], device=kpts_ij.device)
    # return kpts_reduce, inverse_idxs
    return kpts_reduce

def _renormalize_auxbases(auxbases: List[AtomCGTOBasis]) -> List[AtomCGTOBasis]:
    # density basis renormalization, following pyscf here:
    # https://github.com/pyscf/pyscf/blob/7be5e015b2b40181755c71d888449db936604660/pyscf/pbc/df/df.py#L95
    # this renormalization makes the integral of auxbases (not auxbases * auxbases)
    # to be 1

    res: List[AtomCGTOBasis] = []
    # libcint multiply np.sqrt(4*pi) to the basis
    half_sph_norm = 0.5 / np.sqrt(np.pi)
    for atb in auxbases:  # atb is AtomCGTOBasis
        bases: List[CGTOBasis] = []
        for bas in atb.bases:  # bas is CGTOBasis
            assert bas.normalized
            int1 = gaussian_int(bas.angmom * 2 + 2, bas.alphas)
            s = torch.sum(bas.coeffs * int1)
            coeffs = bas.coeffs * (half_sph_norm / s)
            b2 = CGTOBasis(angmom=bas.angmom, coeffs=coeffs, alphas=bas.alphas, normalized=True)
            bases.append(b2)
        atb2 = AtomCGTOBasis(atomz=atb.atomz, bases=bases, pos=atb.pos)
        res.append(atb2)
    return res
