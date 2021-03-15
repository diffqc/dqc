from __future__ import annotations
import os
import re
import copy
import operator
from functools import reduce, lru_cache
import ctypes
from contextlib import contextmanager
from typing import List, Tuple, Optional, Iterator, Callable
import torch
import numpy as np
from dqc.utils.datastruct import AtomCGTOBasis, CGTOBasis

# Terminology:
# * gauss: one gaussian element (multiple gaussian becomes one shell)
# * shell: one contracted basis (the same contracted gaussian for different atoms
#          counted as different shells)
# * ao: shell that has been splitted into its components,
#       e.g. p-shell is splitted into 3 components for cartesian (x, y, z)

NDIM = 3
PTR_RINV_ORIG = 4  # from libcint/src/cint_const.h
BLKSIZE = 128  # same as lib/gto/grid_ao_drv.c

# load the libcint
_curpath = os.path.dirname(os.path.abspath(__file__))
_libcint_path = os.path.join(_curpath, "../../lib/libcint/build/libcint.so")
_libcgto_path = os.path.join(_curpath, "../../lib/libcgto.so")
CINT = ctypes.cdll.LoadLibrary(_libcint_path)
CGTO = ctypes.cdll.LoadLibrary(_libcgto_path)

def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    # get the ctypes of the numpy ndarray
    return a.ctypes.data_as(ctypes.c_void_p)

def int2ctypes(a: int) -> ctypes.c_int:
    # convert the python's integer to ctypes' integer
    return ctypes.c_int(a)

# Optimizer class
class _cintoptHandler(ctypes.c_void_p):
    def __del__(self):
        try:
            CGTO.CINTdel_optimizer(ctypes.byref(self))
        except AttributeError:
            pass

class LibcintWrapper(object):
    def __init__(self, atombases: List[AtomCGTOBasis], spherical: bool = True,
                 basis_normalized: bool = False) -> None:
        self._atombases = atombases
        self._spherical = spherical
        self._basis_normalized = basis_normalized
        self._fracz = False
        self._natoms = len(atombases)

        # get dtype and device for torch's tensors
        self.dtype = atombases[0].bases[0].alphas.dtype
        self.device = atombases[0].bases[0].alphas.device

        # construct _atm, _bas, and _env as well as the parameters
        ptr_env = 20  # initial padding from libcint
        atm_list: List[List[int]] = []
        env_list: List[float] = [0.0] * ptr_env
        bas_list: List[List[int]] = []
        allpos: List[torch.Tensor] = []
        allalphas: List[torch.Tensor] = []
        allcoeffs: List[torch.Tensor] = []
        shell_to_atom: List[int] = []
        ngauss_at_shell: List[int] = []

        # constructing the triplet lists and also collecting the parameters
        nshells = 0
        for iatom, atombasis in enumerate(atombases):
            # construct the atom environment
            assert atombasis.pos.numel() == NDIM, "Please report this bug in Github"
            atomz = atombasis.atomz
            #                charge    ptr_coord, nucl model (unused for standard nucl model)
            atm_list.append([int(atomz), ptr_env, 1, ptr_env + NDIM, 0, 0])
            env_list.extend(atombasis.pos.detach())
            env_list.append(0.0)
            ptr_env += NDIM + 1

            # check if the atomz is fractional
            if isinstance(atomz, float) or \
                    (isinstance(atomz, torch.Tensor) and atomz.is_floating_point()):
                self._fracz = True

            # add the atom position into the parameter list
            # TODO: consider moving allpos into shell
            allpos.append(atombasis.pos.unsqueeze(0))

            nshells += len(atombasis.bases)
            shell_to_atom.extend([iatom] * len(atombasis.bases))

            # then construct the basis
            for shell in atombasis.bases:
                assert shell.alphas.shape == shell.coeffs.shape and shell.alphas.ndim == 1,\
                    "Please report this bug in Github"
                normcoeff = self._normalize_basis(basis_normalized, shell.alphas, shell.coeffs, shell.angmom)
                ngauss = len(shell.alphas)
                #                iatom, angmom,       ngauss, ncontr, kappa, ptr_exp
                bas_list.append([iatom, shell.angmom, ngauss, 1, 0, ptr_env,
                                 # ptr_coeffs,           unused
                                 ptr_env + ngauss, 0])
                env_list.extend(shell.alphas.detach())
                env_list.extend(normcoeff.detach())
                ptr_env += 2 * ngauss

                # add the alphas and coeffs to the parameters list
                allalphas.append(shell.alphas)
                allcoeffs.append(normcoeff)
                ngauss_at_shell.append(ngauss)

        # compile the parameters of this object
        self._allpos_params = torch.cat(allpos, dim=0)  # (natom, NDIM)
        self._allalphas_params = torch.cat(allalphas, dim=0)  # (ntot_gauss)
        self._allcoeffs_params = torch.cat(allcoeffs, dim=0)  # (ntot_gauss)

        # convert the lists to numpy to make it contiguous (Python lists are not contiguous)
        self._atm = np.array(atm_list, dtype=np.int32, order="C")
        self._bas = np.array(bas_list, dtype=np.int32, order="C")
        self._env = np.array(env_list, dtype=np.float64, order="C")

        # construct the full shell mapping
        shell_to_aoloc = [0]
        ao_to_shell: List[int] = []
        ao_to_atom: List[int] = []
        for i in range(nshells):
            nao_at_shell_i = self._nao_at_shell(i)
            shell_to_aoloc_i = shell_to_aoloc[-1] + nao_at_shell_i
            shell_to_aoloc.append(shell_to_aoloc_i)
            ao_to_shell.extend([i] * nao_at_shell_i)
            ao_to_atom.extend([shell_to_atom[i]] * nao_at_shell_i)

        self._ngauss_at_shell_list = ngauss_at_shell
        self._shell_to_aoloc = np.array(shell_to_aoloc, dtype=np.int32)
        self._shell_idxs = (0, nshells)
        self._ao_to_shell = torch.tensor(ao_to_shell, dtype=torch.long, device=self.device)
        self._ao_to_atom = torch.tensor(ao_to_atom, dtype=torch.long, device=self.device)

    @property
    def natoms(self) -> int:
        # return the number of atoms in the environment
        return self._natoms

    @property
    def fracz(self) -> bool:
        # indicating whether we are working with fractional z
        return self._fracz

    @property
    def basis_normalized(self) -> bool:
        return self._basis_normalized

    @property
    def spherical(self) -> bool:
        # returns whether the basis is in spherical coordinate (otherwise, it
        # is in cartesian coordinate)
        return self._spherical

    @property
    def atombases(self) -> List[AtomCGTOBasis]:
        return self._atombases

    @property
    def atm_bas_env(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # returns the triplet lists, i.e. atm, bas, env
        # this shouldn't change in the sliced wrapper
        return self._atm, self._bas, self._env

    @property
    def params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns all the parameters of this object
        # this shouldn't change in the sliced wrapper
        return self._allcoeffs_params, self._allalphas_params, self._allpos_params

    @property
    def shell_idxs(self) -> Tuple[int, int]:
        # returns the lower and upper indices of the shells of this object
        # in the absolute index (upper is exclusive)
        return self._shell_idxs

    @property
    def full_shell_to_aoloc(self) -> np.ndarray:
        # returns the full array mapping from shell index to absolute ao location
        # the atomic orbital absolute index of i-th shell is given by
        # (self.full_shell_to_aoloc[i], self.full_shell_to_aoloc[i + 1])
        # if this object is a subset, then returns the complete mapping
        return self._shell_to_aoloc

    @property
    def full_ao_to_atom(self) -> torch.Tensor:
        # returns the full array mapping from atomic orbital index to the
        # atom location
        return self._ao_to_atom

    @property
    def full_ao_to_shell(self) -> torch.Tensor:
        # returns the full array mapping from atomic orbital index to the
        # shell location
        return self._ao_to_shell

    @property
    def ngauss_at_shell(self) -> List[int]:
        # returns the number of gaussian basis at the given shell
        return self._ngauss_at_shell_list

    @lru_cache(maxsize=32)
    def __len__(self) -> int:
        # total shells
        return self.shell_idxs[-1] - self.shell_idxs[0]

    @lru_cache(maxsize=32)
    def nao(self) -> int:
        # returns the number of atomic orbitals
        shell_idxs = self.shell_idxs
        return self.full_shell_to_aoloc[shell_idxs[-1]] - \
            self.full_shell_to_aoloc[shell_idxs[0]]

    @lru_cache(maxsize=32)
    def ao_idxs(self) -> Tuple[int, int]:
        # returns the lower and upper indices of the atomic orbitals of this object
        # in the full ao map (i.e. absolute indices)
        shell_idxs = self.shell_idxs
        return self.full_shell_to_aoloc[shell_idxs[0]], \
            self.full_shell_to_aoloc[shell_idxs[1]]

    @lru_cache(maxsize=32)
    def ao_to_atom(self) -> torch.Tensor:
        # get the relative mapping from atomic orbital relative index to the
        # absolute atom position
        # this is usually used in scatter in backward calculation
        return self.full_ao_to_atom[slice(*self.ao_idxs())]

    @lru_cache(maxsize=32)
    def ao_to_shell(self) -> torch.Tensor:
        # get the relative mapping from atomic orbital relative index to the
        # absolute shell position
        # this is usually used in scatter in backward calculation
        return self.full_ao_to_shell[slice(*self.ao_idxs())]

    def __getitem__(self, inp) -> LibcintWrapper:
        # get the subset of the shells, but keeping the environment and
        # parameters the same
        assert isinstance(inp, slice)
        assert inp.step is None or inp.step == 1
        assert inp.start is not None or inp.stop is not None

        # complete the slice
        nshells = self.shell_idxs[-1]
        if inp.start is None and inp.stop is not None:
            inp = slice(0, inp.stop)
        elif inp.start is not None and inp.stop is None:
            inp = slice(inp.start, nshells)

        # make the slice positive
        if inp.start < 0:
            inp = slice(inp.start + nshells, inp.stop)
        if inp.stop < 0:
            inp = slice(inp.start, inp.stop + nshells)

        return SubsetLibcintWrapper(self, inp)

    @lru_cache(maxsize=32)
    def get_uncontracted_wrapper(self) -> Tuple[LibcintWrapper, torch.Tensor]:
        # returns the uncontracted LibcintWrapper as well as the mapping from
        # uncontracted atomic orbital (relative index) to the relative index
        # of the atomic orbital
        new_atombases = []
        for atombasis in self.atombases:
            atomz = atombasis.atomz
            pos = atombasis.pos
            new_bases = []
            for shell in atombasis.bases:
                angmom = shell.angmom
                alphas = shell.alphas
                coeffs = shell.coeffs
                new_bases.extend([
                    CGTOBasis(angmom, alpha[None], coeff[None]) for (alpha, coeff) in zip(alphas, coeffs)
                ])
            new_atombases.append(AtomCGTOBasis(atomz=atomz, bases=new_bases, pos=pos))
        uncontr_wrapper = LibcintWrapper(
            new_atombases, spherical=self.spherical,
            basis_normalized=self.basis_normalized)

        # get the mapping uncontracted ao to the contracted ao
        uao2ao: List[int] = []
        idx_ao = 0
        # iterate over shells
        for i in range(len(self)):
            nao = self._nao_at_shell(i)
            uao2ao += list(range(idx_ao, idx_ao + nao)) * self.ngauss_at_shell[i]
            idx_ao += nao
        uao2ao_res = torch.tensor(uao2ao, dtype=torch.long, device=self.device)
        return uncontr_wrapper, uao2ao_res

    # integrals
    def int1e(self, shortname: str, other: Optional[LibcintWrapper] = None, *,
              # additional options for some specific integrals
              rinv_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 2-centre 1-electron integral

        # check and set the other parameters
        other1 = self._check_and_set(other)

        # set the rinv_pos arguments
        if "rinv" in shortname:
            assert isinstance(rinv_pos, torch.Tensor), "The keyword rinv_pos must be specified"
        else:
            # don't really care, it will be ignored
            rinv_pos = torch.zeros(1, dtype=self.dtype, device=self.device)

        return _Int2cFunction.apply(*self.params,
                                    rinv_pos,
                                    [self, other1],
                                    "int1e", shortname)

    def int2e(self, shortname: str,
              other1: Optional[LibcintWrapper] = None,
              other2: Optional[LibcintWrapper] = None,
              other3: Optional[LibcintWrapper] = None) -> torch.Tensor:
        # 4-centre 2-electron integral

        # check and set the others
        other1w = self._check_and_set(other1)
        other2w = self._check_and_set(other2)
        other3w = self._check_and_set(other3)
        return _Int4cFunction.apply(
            *self.params,
            [self, other1w, other2w, other3w],
            "int2e", shortname)

    # evaluation of the gaussian basis
    def evl(self, shortname: str, rgrid: torch.Tensor) -> torch.Tensor:
        # expand ao_to_atom to have shape of (nao, ndim)
        ao_to_atom = self.ao_to_atom().unsqueeze(-1).expand(-1, NDIM)

        # rgrid: (ngrid, ndim)
        return _EvalGTO.apply(
            # tensors
            *self.params, rgrid,

            # nontensors or int tensors
            ao_to_atom, self, shortname)

    # shortcuts
    def overlap(self, other: Optional[LibcintWrapper] = None) -> torch.Tensor:
        return self.int1e("ovlp", other=other)

    def kinetic(self, other: Optional[LibcintWrapper] = None) -> torch.Tensor:
        return self.int1e("kin", other=other)

    def nuclattr(self, other: Optional[LibcintWrapper] = None) -> torch.Tensor:
        if not self.fracz:  # ???
            return self.int1e("nuc", other=other)
        else:
            res = torch.tensor([])
            allpos_params = self.params[-1]
            for i in range(self.natoms):
                y = self.int1e("rinv", other=other, rinv_pos=allpos_params[i]) * \
                    (-self._atombases[i].atomz)
                res = y if (i == 0) else (res + y)
            return res

    def elrep(self,
              other1: Optional[LibcintWrapper] = None,
              other2: Optional[LibcintWrapper] = None,
              other3: Optional[LibcintWrapper] = None,
              ) -> torch.Tensor:
        return self.int2e("ar12b", other1, other2, other3)

    def eval_gto(self, rgrid: torch.Tensor) -> torch.Tensor:
        # rgrid: (ngrid, ndim)
        # return: (nao, ngrid)
        return self.evl("", rgrid)

    def eval_gradgto(self, rgrid: torch.Tensor) -> torch.Tensor:
        # rgrid: (ngrid, ndim)
        # return: (ndim, nao, ngrid)
        return self.evl("ip", rgrid)

    def eval_laplgto(self, rgrid: torch.Tensor) -> torch.Tensor:
        # rgrid: (ngrid, ndim)
        # return: (nao, ngrid)
        return self.evl("lapl", rgrid)

    ############### misc functions ###############
    @contextmanager
    def centre_on_r(self, r: torch.Tensor) -> Iterator:
        # set the centre of coordinate to r (usually used in rinv integral)
        # r: (ndim,)
        try:
            env = self.atm_bas_env[-1]
            prev_centre = env[PTR_RINV_ORIG: PTR_RINV_ORIG + NDIM]
            env[PTR_RINV_ORIG: PTR_RINV_ORIG + NDIM] = r.detach().numpy()
            yield
        finally:
            env[PTR_RINV_ORIG: PTR_RINV_ORIG + NDIM] = prev_centre

    ############### private functions ###################
    def _check_and_set(self, other: Optional[LibcintWrapper]) -> LibcintWrapper:
        # check the value and set the default value of "other" in the integrals
        if other is not None:
            atm0, bas0, env0 = self.atm_bas_env
            atm1, bas1, env1 = other.atm_bas_env
            assert id(atm0) == id(atm1)
            assert id(bas0) == id(bas1)
            assert id(env0) == id(env1)
        else:
            other = self
        assert isinstance(other, LibcintWrapper)
        return other

    def _normalize_basis(self, basis_normalized: bool, alphas: torch.Tensor,
                         coeffs: torch.Tensor, angmom: int) -> torch.Tensor:
        # the normalization is obtained from CINTgto_norm from
        # libcint/src/misc.c, or
        # https://github.com/sunqm/libcint/blob/b8594f1d27c3dad9034984a2a5befb9d607d4932/src/misc.c#L80

        # if the basis has been normalized before, then just return the coeffs
        if basis_normalized:
            return coeffs

        # precomputed factor:
        # 2 ** (2 * angmom + 3) * factorial(angmom + 1) * / \
        # (factorial(angmom * 2 + 2) * np.sqrt(np.pi)))
        factor = [
            2.256758334191025,  # 0
            1.5045055561273502,  # 1
            0.6018022224509401,  # 2
            0.17194349212884005,  # 3
            0.03820966491752001,  # 4
            0.006947211803185456,  # 5
            0.0010688018158746854,  # 6
        ]
        return coeffs * torch.sqrt(factor[angmom] * (2 * alphas) ** (angmom + 1.5))

    def _nao_at_shell(self, sh: int) -> int:
        # returns the number of atomic orbital at the given shell index
        if self.spherical:
            op = CINT.CINTcgto_spheric
        else:
            op = CINT.CINTcgto_cart
        bas = self.atm_bas_env[1]
        return op(int2ctypes(sh), np2ctypes(bas))

class SubsetLibcintWrapper(LibcintWrapper):
    """
    A class to represent the subset of LibcintWrapper.
    If put into integrals or evaluations, this class will only evaluate
        the subset of the shells from its parent.
    The environment will still be the same as its parent.
    """
    def __init__(self, parent: LibcintWrapper, subset: slice):
        self._parent = parent
        self._shell_idxs = subset.start, subset.stop

    @property
    def shell_idxs(self) -> Tuple[int, int]:
        return self._shell_idxs

    @lru_cache(maxsize=32)
    def get_uncontracted_wrapper(self):
        # returns the uncontracted LibcintWrapper as well as the mapping from
        # uncontracted atomic orbital (relative index) to the relative index
        # of the atomic orbital of the contracted wrapper

        pu_wrapper, p_uao2ao = self._parent.get_uncontracted_wrapper()

        # determine the corresponding shell indices in the new uncontracted wrapper
        shell_idxs = self.shell_idxs
        gauss_idx0 = sum(self._parent.ngauss_at_shell[: shell_idxs[0]])
        gauss_idx1 = sum(self._parent.ngauss_at_shell[: shell_idxs[1]])
        u_wrapper = pu_wrapper[gauss_idx0: gauss_idx1]

        # construct the uao (relative index) mapping to the absolute index
        # of the atomic orbital in the contracted basis
        uao2ao = []
        idx_ao = 0
        for i in range(shell_idxs[0], shell_idxs[1]):
            nao = self._parent._nao_at_shell(i)
            uao2ao += list(range(idx_ao, idx_ao + nao)) * self._parent.ngauss_at_shell[i]
            idx_ao += nao
        uao2ao_res = torch.tensor(uao2ao, dtype=torch.long, device=self.device)
        return u_wrapper, uao2ao_res

    def __getitem__(self, inp):
        raise NotImplementedError("Indexing of SubsetLibcintWrapper is not implemented")

    def __getattr__(self, name):
        return getattr(self._parent, name)

############### autograd functions ###############
class _Int2cFunction(torch.autograd.Function):
    # wrapper class to provide the gradient of the 2-centre integrals
    @staticmethod
    def forward(ctx,  # type: ignore
                allcoeffs: torch.Tensor, allalphas: torch.Tensor, allposs: torch.Tensor,
                rinv_pos: torch.Tensor,
                wrappers: List[LibcintWrapper], int_type: str, shortname: str) -> torch.Tensor:
        # allcoeffs: (ngauss_tot,)
        # allalphas: (ngauss_tot,)
        # allposs: (natom, ndim)
        # rinv_pos: (ndim,) if contains "rinv"
        #           rinv_pos is only meaningful if shortname contains "rinv"
        # In "rinv", rinv_pos becomes the centre
        # Wrapper0 and wrapper1 must have the same _atm, _bas, and _env.
        # The check should be done before calling this function.
        # Those tensors are not used directly in the forward calculation, but
        #   required for backward propagation
        assert len(wrappers) == 2

        if "rinv" in shortname:
            assert rinv_pos.ndim == 1 and rinv_pos.shape[0] == NDIM
            with wrappers[0].centre_on_r(rinv_pos):
                out_tensor = Intor(int_type, shortname, wrappers).calc()
        else:
            out_tensor = Intor(int_type, shortname, wrappers).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs,
                              rinv_pos)
        ctx.other_info = (wrappers, int_type, shortname)
        return out_tensor  # (..., nao0, nao1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        # grad_out: (..., nao0, nao1)
        allcoeffs, allalphas, allposs, \
            rinv_pos = ctx.saved_tensors
        wrappers, int_type, shortname = ctx.other_info

        # gradient for all atomic positions
        grad_allposs: Optional[torch.Tensor] = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = grad_allposs.transpose(-2, -1)  # (ndim, natom)

            # get the integrals required for the derivatives
            sname_derivs = [_get_intgl_deriv_shortname(int_type, shortname, s) for s in ("r1", "r2")]
            int_fcn = lambda wrappers, name: _Int2cFunction.apply(
                *ctx.saved_tensors, wrappers, int_type, name)
            # list of tensors with shape: (ndim, ..., nao0, nao1)
            dout_dposs = _get_integrals(sname_derivs, wrappers, int_type, int_fcn)

            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *dout_dposs[0].shape[-2:])
            grad_out2 = grad_out.reshape(shape[1:])
            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            grad_dpos_i = -torch.einsum("sij,dsij->di", grad_out2, dout_dposs[0].reshape(shape))
            grad_dpos_j = -torch.einsum("sij,dsij->dj", grad_out2, dout_dposs[1].reshape(shape))

            # grad_allpossT is only a view of grad_allposs, so the operation below
            # also changes grad_allposs
            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom0, src=grad_dpos_i)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom1, src=grad_dpos_j)

            grad_allposs_nuc = torch.zeros_like(grad_allposs)
            if "nuc" in shortname:
                # allposs: (natoms, ndim)
                natoms = allposs.shape[0]
                sname_rinv = shortname.replace("nuc", "rinv")
                sname_derivs = [_get_intgl_deriv_shortname(int_type, sname_rinv, s) for s in ("r1", "r2")]

                for i in range(natoms):
                    atomz = wrappers[0].atombases[i].atomz

                    # get the integrals
                    int_fcn = lambda wrappers, name: _Int2cFunction.apply(
                        allcoeffs, allalphas, allposs, allposs[i],
                        wrappers, int_type, name)
                    dout_datposs = _get_integrals(sname_derivs, wrappers, int_type, int_fcn)  # (ndim, ..., nao, nao)

                    grad_datpos = grad_out * (dout_datposs[0] + dout_datposs[1])
                    grad_datpos = grad_datpos.reshape(grad_datpos.shape[0], -1).sum(dim=-1)
                    grad_allposs_nuc[i] = (-atomz) * grad_datpos

                grad_allposs += grad_allposs_nuc

        # gradient for the rinv_pos in rinv integral
        grad_rinv_pos: Optional[torch.Tensor] = None
        if rinv_pos.requires_grad and "rinv" in shortname:
            # rinv_pos: (ndim)
            # get the integrals for the derivatives
            sname_derivs = [_get_intgl_deriv_shortname(int_type, shortname, s) for s in ("r1", "r2")]
            int_fcn = lambda wrappers, name: _Int2cFunction.apply(
                *ctx.saved_tensors, wrappers, int_type, name)
            dout_datposs = _get_integrals(sname_derivs, wrappers, int_type, int_fcn)

            grad_datpos = grad_out * (dout_datposs[0] + dout_datposs[1])
            grad_rinv_pos = grad_datpos.reshape(grad_datpos.shape[0], -1).sum(dim=-1)

        # gradient for the basis coefficients
        grad_allcoeffs: Optional[torch.Tensor] = None
        grad_allalphas: Optional[torch.Tensor] = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper and mapping
            # uao2aos: list of (nu_ao0,), (nu_ao1,)
            u_wrappers_tup, uao2aos_tup = zip(*[w.get_uncontracted_wrapper() for w in wrappers])
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # get the uncontracted (gathered) grad_out
            u_grad_out = _gather_at_dims(grad_out, mapidxs=uao2aos, dims=[-2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()
            ao2shl1 = u_wrappers[1].ao_to_shell()

            # calculate the gradient w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)  # (ngauss)

                # get the uncontracted version of the integral
                dout_dcoeff = _Int2cFunction.apply(
                    *u_params, rinv_pos, u_wrappers, int_type, shortname)  # (..., nu_ao0, nu_ao1)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1, index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1, index=ao2shl1)  # (nu_ao1)
                # divide done here instead of after scatter to make the 2nd gradient
                # calculation correct.
                # division can also be done after scatter for more efficient 1st grad
                # calculation, but it gives the wrong result for 2nd grad
                dout_dcoeff_i = dout_dcoeff / coeffs_ao0[:, None]
                dout_dcoeff_j = dout_dcoeff / coeffs_ao1

                # (nu_ao)
                grad_dcoeff_i = torch.einsum("...ij,...ij->i", u_grad_out, dout_dcoeff_i)
                grad_dcoeff_j = torch.einsum("...ij,...ij->j", u_grad_out, dout_dcoeff_j)
                # grad_dcoeff = grad_dcoeff_i + grad_dcoeff_j

                # scatter the grad
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl0, src=grad_dcoeff_i)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl1, src=grad_dcoeff_j)

            # calculate the gradient w.r.t. alphas
            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                u_int_fcn = lambda u_wrappers, name: _Int2cFunction.apply(
                    *u_params, rinv_pos, u_wrappers, int_type, name)

                # get the uncontracted integrals
                sname_derivs = [_get_intgl_deriv_shortname(int_type, shortname, s) for s in ("a1", "a2")]
                dout_dalphas = _get_integrals(sname_derivs, u_wrappers, int_type, u_int_fcn)

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_dalpha_i = -torch.einsum("...ij,...ij->i", u_grad_out, dout_dalphas[0])
                grad_dalpha_j = -torch.einsum("...ij,...ij->j", u_grad_out, dout_dalphas[1])
                # grad_dalpha = (grad_dalpha_i + grad_dalpha_j)  # (nu_ao)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl0, src=grad_dalpha_i)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl1, src=grad_dalpha_j)

        return grad_allcoeffs, grad_allalphas, grad_allposs, \
            grad_rinv_pos, \
            None, None, None

class _Int4cFunction(torch.autograd.Function):
    # wrapper class for the 4-centre integrals
    @staticmethod
    def forward(ctx,  # type: ignore
                allcoeffs: torch.Tensor, allalphas: torch.Tensor, allposs: torch.Tensor,
                wrappers: List[LibcintWrapper],
                int_type: str, shortname: str) -> torch.Tensor:

        assert len(wrappers) == 4

        out_tensor = Intor(int_type, shortname, wrappers).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs)
        ctx.other_info = (wrappers, int_type, shortname)
        return out_tensor  # (..., nao0, nao1, nao2, nao3)

    @staticmethod
    def backward(ctx, grad_out) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        # grad_out: (..., nao0, nao1, nao2, nao3)
        allcoeffs, allalphas, allposs = ctx.saved_tensors
        wrappers, int_type, shortname = ctx.other_info
        naos = grad_out.shape[-4:]

        # calculate the gradient w.r.t. positions
        grad_allposs: Optional[torch.Tensor] = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = grad_allposs.transpose(-2, -1)  # (ndim, natom)

            sname_derivs = [_get_intgl_deriv_shortname(int_type, shortname, sname)
                            for sname in ("ra1", "ra2", "rb1", "rb2")]
            int_fcn = lambda wrappers, name: _Int4cFunction.apply(
                *ctx.saved_tensors, wrappers, int_type, name)
            dout_dposs = _get_integrals(sname_derivs, wrappers, int_type, int_fcn)

            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *naos)
            grad_out2 = grad_out.reshape(*shape[1:])
            grad_pos_a1 = -torch.einsum("dzijkl,zijkl->di", dout_dposs[0].reshape(*shape), grad_out2)
            grad_pos_a2 = -torch.einsum("dzijkl,zijkl->dj", dout_dposs[1].reshape(*shape), grad_out2)
            grad_pos_b1 = -torch.einsum("dzijkl,zijkl->dk", dout_dposs[2].reshape(*shape), grad_out2)
            grad_pos_b2 = -torch.einsum("dzijkl,zijkl->dl", dout_dposs[3].reshape(*shape), grad_out2)

            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            ao_to_atom2 = wrappers[2].ao_to_atom().expand(ndim, -1)
            ao_to_atom3 = wrappers[3].ao_to_atom().expand(ndim, -1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom0, src=grad_pos_a1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom1, src=grad_pos_a2)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom2, src=grad_pos_b1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom3, src=grad_pos_b2)

        # gradients for the basis coefficients
        grad_allcoeffs: Optional[torch.Tensor] = None
        grad_allalphas: Optional[torch.Tensor] = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper, and expanded grad_out
            # uao2ao: (nu_ao)
            u_wrappers_tup, uao2aos_tup = zip(*[w.get_uncontracted_wrapper() for w in wrappers])
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # u_grad_out: (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
            u_grad_out = _gather_at_dims(grad_out, mapidxs=uao2aos, dims=[-4, -3, -2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()  # (nu_ao0,)
            ao2shl1 = u_wrappers[1].ao_to_shell()
            ao2shl2 = u_wrappers[2].ao_to_shell()
            ao2shl3 = u_wrappers[3].ao_to_shell()

            # calculate the grad w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)

                # (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
                dout_dcoeff = _Int4cFunction.apply(*u_params, u_wrappers, int_type, shortname)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1, index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1, index=ao2shl1)  # (nu_ao1)
                coeffs_ao2 = torch.gather(allcoeffs, dim=-1, index=ao2shl2)  # (nu_ao2)
                coeffs_ao3 = torch.gather(allcoeffs, dim=-1, index=ao2shl3)  # (nu_ao3)
                # dout_dcoeff_*: (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
                dout_dcoeff_a1 = dout_dcoeff / coeffs_ao0[:, None, None, None]
                dout_dcoeff_a2 = dout_dcoeff / coeffs_ao1[:, None, None]
                dout_dcoeff_b1 = dout_dcoeff / coeffs_ao2[:, None]
                dout_dcoeff_b2 = dout_dcoeff / coeffs_ao3

                # reduce the uncontracted integrations
                # grad_coeff_*: (nu_ao*)
                grad_coeff_a1 = torch.einsum("...ijkl,...ijkl->i", dout_dcoeff_a1, u_grad_out)
                grad_coeff_a2 = torch.einsum("...ijkl,...ijkl->j", dout_dcoeff_a2, u_grad_out)
                grad_coeff_b1 = torch.einsum("...ijkl,...ijkl->k", dout_dcoeff_b1, u_grad_out)
                grad_coeff_b2 = torch.einsum("...ijkl,...ijkl->l", dout_dcoeff_b2, u_grad_out)
                # grad_coeff_all = grad_coeff_a1 + grad_coeff_a2 + grad_coeff_b1 + grad_coeff_b2

                # scatter to the coefficients
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl0, src=grad_coeff_a1)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl1, src=grad_coeff_a2)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl2, src=grad_coeff_b1)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl3, src=grad_coeff_b2)

            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                # get the uncontracted integrals
                sname_derivs = [_get_intgl_deriv_shortname(int_type, shortname, sname)
                                for sname in ("aa1", "aa2", "ab1", "ab2")]
                u_int_fcn = lambda u_wrappers, name: _Int4cFunction.apply(
                    *u_params, u_wrappers, int_type, name)
                dout_dalphas = _get_integrals(sname_derivs, u_wrappers, int_type, u_int_fcn)

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_alpha_a1 = -torch.einsum("...ijkl,...ijkl->i", dout_dalphas[0], u_grad_out)
                grad_alpha_a2 = -torch.einsum("...ijkl,...ijkl->j", dout_dalphas[1], u_grad_out)
                grad_alpha_b1 = -torch.einsum("...ijkl,...ijkl->k", dout_dalphas[2], u_grad_out)
                grad_alpha_b2 = -torch.einsum("...ijkl,...ijkl->l", dout_dalphas[3], u_grad_out)
                # grad_alpha_all = (grad_alpha_a1 + grad_alpha_a2 + grad_alpha_b1 + grad_alpha_b2)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl0, src=grad_alpha_a1)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl1, src=grad_alpha_a2)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl2, src=grad_alpha_b1)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl3, src=grad_alpha_b2)

        return grad_allcoeffs, grad_allalphas, grad_allposs, \
            None, None, None

class _EvalGTO(torch.autograd.Function):
    # wrapper class to provide the gradient for evaluating the contracted gto
    @staticmethod
    def forward(ctx,  # type: ignore
                # tensors not used in calculating the forward, but required
                # for the backward propagation
                alphas: torch.Tensor,  # (ngauss_tot)
                coeffs: torch.Tensor,  # (ngauss_tot)
                pos: torch.Tensor,  # (natom, ndim)

                # tensors used in forward
                rgrid: torch.Tensor,  # (ngrid, ndim)

                # other non-tensor info
                ao_to_atom: torch.Tensor,  # int tensor (nao, ndim)
                wrapper: LibcintWrapper,
                shortname: str) -> torch.Tensor:

        res = gto_evaluator(wrapper, shortname, rgrid)  # (*, nao, ngrid)
        ctx.save_for_backward(alphas, coeffs, pos, rgrid)
        ctx.other_info = (ao_to_atom, wrapper, shortname)
        return res

    @staticmethod
    def backward(ctx, grad_res: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        # grad_res: (*, nao, ngrid)
        ao_to_atom, wrapper, shortname = ctx.other_info
        alphas, coeffs, pos, rgrid = ctx.saved_tensors

        # TODO: implement the gradient w.r.t. alphas and coeffs
        grad_alphas = None
        grad_coeffs = None

        # calculate the gradient w.r.t. basis' pos and rgrid
        grad_pos = None
        grad_rgrid = None
        if rgrid.requires_grad or pos.requires_grad:
            opsname = _get_evalgto_derivname(shortname, "r")
            dresdr = _EvalGTO.apply(*ctx.saved_tensors,
                                    ao_to_atom, wrapper, opsname)  # (ndim, *, nao, ngrid)
            grad_r = dresdr * grad_res  # (ndim, *, nao, ngrid)

            if rgrid.requires_grad:
                grad_rgrid = grad_r.reshape(dresdr.shape[0], -1, dresdr.shape[-1])
                grad_rgrid = grad_rgrid.sum(dim=1).transpose(-2, -1)  # (ngrid, ndim)

            if pos.requires_grad:
                grad_rao = torch.movedim(grad_r, -2, 0)  # (nao, ndim, *, ngrid)
                grad_rao = -grad_rao.reshape(*grad_rao.shape[:2], -1).sum(dim=-1)  # (nao, ndim)
                grad_pos = torch.zeros_like(pos)  # (natom, ndim)
                grad_pos.scatter_add_(dim=0, index=ao_to_atom, src=grad_rao)

        return grad_alphas, grad_coeffs, grad_pos, grad_rgrid, \
            None, None, None, None, None

################### integrator (direct interface to libcint) ###################
class Intor(object):
    def __init__(self, int_type: str, shortname: str, wrappers: List[LibcintWrapper]):
        assert len(wrappers) > 0
        wrapper0 = wrappers[0]
        self.int_type = int_type
        self.atm, self.bas, self.env = wrapper0.atm_bas_env
        self.wrapper0 = wrapper0

        # get the operator
        opname = _get_intgl_name(int_type, shortname, wrapper0.spherical)
        self.op = getattr(CINT, opname)
        self.optimizer = _get_intgl_optimizer(opname, self.atm, self.bas, self.env)

        # prepare the output
        comp_shape = _get_intgl_components_shape(shortname)
        self.outshape = comp_shape + tuple(w.nao() for w in wrappers)
        self.ncomp = reduce(operator.mul, comp_shape, 1)
        self.shls_slice = sum((w.shell_idxs for w in wrappers), ())
        self.integral_done = False

    def calc(self) -> torch.Tensor:
        assert not self.integral_done
        self.integral_done = True
        if self.int_type == "int1e":
            return self._int2c()
        elif self.int_type == "int2e":
            return self._int4c()
        else:
            raise ValueError("Unknown integral type: %s" % self.int_type)

    def _int2c(self):
        # performing 2-centre integrals with libcint
        drv = CGTO.GTOint2c
        outshape = self.outshape
        out = np.empty((*outshape[:-2], outshape[-1], outshape[-2]), dtype=np.float64)
        drv(self.op,
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.ncomp),
            ctypes.c_int(0),  # do not assume hermitian
            (ctypes.c_int * len(self.shls_slice))(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc),
            self.optimizer,
            np2ctypes(self.atm), int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas), int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env))

        out = np.swapaxes(out, -2, -1)
        # TODO: check if we need to do the lines below for 3rd order grad and higher
        # if out.ndim > 2:
        #     out = np.moveaxis(out, -3, 0)
        out_tensor = torch.as_tensor(out, dtype=self.wrapper0.dtype,
                                     device=self.wrapper0.device)
        return out_tensor

    def _int4c(self):
        # performing 4-centre integrals with libcint
        out = np.empty(self.outshape, dtype=np.float64)
        drv = CGTO.GTOnr2e_fill_drv
        fill = CGTO.GTOnr2e_fill_s1
        prescreen = ctypes.POINTER(ctypes.c_void_p)()
        drv(self.op, fill, prescreen,
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.ncomp),
            (ctypes.c_int * 8)(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc),
            self.optimizer,
            np2ctypes(self.atm), int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas), int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env))

        out_tensor = torch.as_tensor(out, dtype=self.wrapper0.dtype,
                                     device=self.wrapper0.device)
        return out_tensor

def _get_intgl_name(int_type: str, shortname: str, spherical: bool) -> str:
    # convert the shortname into full name of the integral in libcint
    suffix = ("_" + shortname) if shortname != "" else shortname
    cartsph = "sph" if spherical else "cart"
    return "%s%s_%s" % (int_type, suffix, cartsph)

def _get_intgl_optimizer(opname: str,
                         atm: np.ndarray, bas: np.ndarray, env: np.ndarray)\
                         -> ctypes.c_void_p:
    # get the optimizer of the integrals
    # setup the optimizer
    cintopt = ctypes.POINTER(ctypes.c_void_p)()
    optname = opname.replace("_cart", "").replace("_sph", "") + "_optimizer"
    copt = getattr(CINT, optname)
    copt(ctypes.byref(cintopt),
         np2ctypes(atm), int2ctypes(atm.shape[0]),
         np2ctypes(bas), int2ctypes(bas.shape[0]),
         np2ctypes(env))
    opt = ctypes.cast(cintopt, _cintoptHandler)
    return opt

def _get_intgl_components_shape(shortname: str) -> Tuple[int, ...]:
    # returns the component shape of the array of the given integral

    # calculate the occurence of a pattern in string s
    re_pattern = r"({pattern})".format(pattern="ip")
    n_ip = len(re.findall(re_pattern, shortname))

    comp_shape = (NDIM, ) * n_ip
    return comp_shape

############### name derivation manager functions ###############
def _get_intgl_deriv_shortname(int_type: str, shortname: str, derivmode: str) -> str:
    # get the operation required for the derivation of the integration operator

    # get the _insert_pattern function
    if int_type == "int1e":
        def _insert_pattern(shortname: str, derivmode: str, pattern: str) -> str:
            if derivmode == "1":
                return "%s%s" % (pattern, shortname)
            elif derivmode == "2":
                return "%s%s" % (shortname, pattern)
            else:
                raise RuntimeError("Unknown derivmode: %s" % derivmode)
    elif int_type == "int2e":
        def _insert_pattern(shortname: str, derivmode: str, pattern: str) -> str:
            if derivmode == "a1":
                return "%s%s" % (pattern, shortname)
            elif derivmode == "a2":
                # insert after the first "a"
                idx_a = shortname.find("a")
                return shortname[:idx_a + 1] + pattern + shortname[idx_a + 1:]
            elif derivmode == "b1":
                # insert before the last "b"
                idx_b = shortname.rfind("b")
                return shortname[:idx_b] + pattern + shortname[idx_b:]
            elif derivmode == "b2":
                return "%s%s" % (shortname, pattern)
            else:
                raise RuntimeError("Unknown derivmode: %s" % derivmode)
    else:
        raise ValueError("Unknown integral type: %s" % int_type)

    if derivmode.startswith("r"):
        return _insert_pattern(shortname, derivmode[1:], "ip")
    elif derivmode.startswith("a"):
        return _insert_pattern(shortname, derivmode[1:], "rr")
    else:
        raise RuntimeError("Unknown derivmode: %s" % derivmode)

def _get_integrals(int_names: List[str],
                   wrappers: List[LibcintWrapper],
                   int_type: str,
                   int_fcn: Callable[[List[LibcintWrapper], str], torch.Tensor]) \
                   -> List[torch.Tensor]:
    # return the list of tensors of the integrals given by the list of integral names.
    # int_fcn is the integral function that receives the name and returns the results.

    res: List[torch.Tensor] = []
    # indicating if the integral is available in the libcint-generated file
    int_avail: List[bool] = [False] * len(int_names)

    for i in range(len(int_names)):
        res_i: Optional[torch.Tensor] = None

        # check if the integral can be calculated from the previous results
        for j in range(i - 1, -1, -1):

            # check the integral names equivalence
            transpose_path = _intgl_shortname_equiv(int_names[j], int_names[i], int_type)
            if transpose_path is not None:

                # if the swapped wrappers remain unchanged, then just use the
                # transposed version of the previous version
                # TODO: think more about this (do we need to use different
                # transpose path? e.g. transpose_path[::-1])
                twrappers = _swap_list(wrappers, transpose_path)
                if twrappers == wrappers:
                    res_i = _transpose(res[j], transpose_path)
                    break

                # otherwise, use the swapped integral with the swapped wrappers,
                # only if the integral is available in the libcint-generated
                # files
                elif int_avail[j]:
                    res_i = int_fcn(twrappers, int_names[j])
                    res_i = _transpose(res_i, transpose_path)
                    break

                # if the integral is not available, then continue the searching
                else:
                    continue

        if res_i is None:
            # successfully executing the line below indicates that the integral
            # is available in the libcint-generated files
            res_i = int_fcn(wrappers, int_names[i])
            int_avail[i] = True

        res.append(res_i)

    return res

def _intgl_shortname_equiv(s0: str, s1: str, int_type: str) -> Optional[List[Tuple[int, int]]]:
    # check if the integration s1 can be achieved by transposing s0
    # returns None if it cannot.
    # returns the list of two dims if it can for the transpose-path of s0
    # to get the same result as s1

    if int_type == "int1e":
        patterns = ["nuc", "ovlp", "rinv", "kin"]
        transpose_paths = [
            [],
            [(-1, -2)],
        ]
    elif int_type == "int2e":
        patterns = ["r12", "a", "b"]
        transpose_paths = [
            [],
            [(-3, -4)],
            [(-1, -2)],
            [(-1, -3), (-2, -4)],
            [(-1, -3), (-2, -4), (-2, -1)],
            [(-1, -3), (-2, -4), (-3, -4)],
        ]
    else:
        raise ValueError("Unknown integral type: %s" % int_type)

    return _intgl_shortname_equiv_helper(s0, s1, patterns, transpose_paths)

def _intgl_shortname_equiv_helper(s0: str, s1: str, patterns: List[str],
                                  transpose_paths: List) -> Optional[List[Tuple[int, int]]]:
    # find the transpose path to get the s1 integral from s0.
    # this function should return the transpose path from s0 to reach s1.
    # returns None if it is not possible.

    def _parse_pattern(s: str, patterns: List[str]) -> List[str]:
        for c in patterns:
            s = s.replace(c, "|")
        return s.split("|")

    p0 = _parse_pattern(s0, patterns)
    p1 = _parse_pattern(s1, patterns)

    def _swap(p: List[str], path: List[Tuple[int, int]]):
        # swap the pattern according to the given transpose path
        r = p[:]  # make a copy
        for i0, i1 in path:
            r[i0], r[i1] = r[i1], r[i0]
        return r

    for transpose_path in transpose_paths:
        if _swap(p0, transpose_path) == p1:
            return transpose_path
    return None

def _transpose(a: torch.Tensor, axes: List[Tuple[int, int]]) -> torch.Tensor:
    # perform the transpose of two axes for tensor a
    for axis2 in axes:
        a = a.transpose(*axis2)
    return a

def _swap_list(a: List, swaps: List[Tuple[int, int]]) -> List:
    # swap the elements according to the swaps input
    res = copy.copy(a)  # shallow copy
    for idxs in swaps:
        res[idxs[0]], res[idxs[1]] = res[idxs[1]], res[idxs[0]]  # swap the elements
    return res

def _gather_at_dims(inp: torch.Tensor, mapidxs: List[torch.Tensor],
                    dims: List[int]) -> torch.Tensor:
    # expand inp in the dimension dim by gathering values based on the given
    # mapping indices

    # mapidx: (nnew,) with value from 0 to nold - 1
    # inp: (..., nold, ...)
    # out: (..., nnew, ...)
    out = inp
    for (dim, mapidx) in zip(dims, mapidxs):
        if dim < 0:
            dim = out.ndim + dim
        map2 = mapidx[(...,) + (None,) * (out.ndim - 1 - dim)]
        map2 = map2.expand(*out.shape[:dim], -1, *out.shape[dim + 1:])
        out = torch.gather(out, dim=dim, index=map2)
    return out

################### evaluator (direct interfact to libcgto) ###################
def gto_evaluator(wrapper: LibcintWrapper, shortname: str, rgrid: torch.Tensor):
    # NOTE: this function do not propagate gradient and should only be used
    # in this file only

    # rgrid: (ngrid, ndim)
    # returns: (*, nao, ngrid)

    ngrid = rgrid.shape[0]
    nshells = len(wrapper)
    nao = wrapper.nao()
    opname = _get_evalgto_opname(shortname, wrapper.spherical)
    outshape = _get_evalgto_compshape(shortname) + (nao, ngrid)

    out = np.empty(outshape, dtype=np.float64)
    non0tab = np.ones(((ngrid + BLKSIZE - 1) // BLKSIZE, nshells),
                      dtype=np.int8)

    # TODO: check if we need to transpose it first?
    rgrid = rgrid.contiguous()
    coords = np.asarray(rgrid, dtype=np.float64, order='F')
    ao_loc = np.asarray(wrapper.full_shell_to_aoloc, dtype=np.int32)

    c_shls = (ctypes.c_int * 2)(*wrapper.shell_idxs)
    c_ngrid = ctypes.c_int(ngrid)

    # evaluate the orbital
    operator = getattr(CGTO, opname)
    operator.restype = ctypes.c_double
    atm, bas, env = wrapper.atm_bas_env
    operator(c_ngrid, c_shls,
             np2ctypes(ao_loc),
             np2ctypes(out),
             np2ctypes(coords),
             np2ctypes(non0tab),
             np2ctypes(atm), int2ctypes(atm.shape[0]),
             np2ctypes(bas), int2ctypes(bas.shape[0]),
             np2ctypes(env))

    out_tensor = torch.tensor(out, dtype=wrapper.dtype, device=wrapper.device)
    return out_tensor

def _get_evalgto_opname(shortname: str, spherical: bool) -> str:
    # returns the complete name of the evalgto operation
    sname = ("_" + shortname) if (shortname != "") else ""
    suffix = "_sph" if spherical else "_cart"
    return "GTOval%s%s" % (sname, suffix)

def _get_evalgto_compshape(shortname: str) -> Tuple[int, ...]:
    # returns the component shape of the evalgto function

    # count "ip" only at the beginning
    n_ip = len(re.findall(r"^(?:ip)*(?:ip)?", shortname)[0]) // 2
    return (NDIM, ) * n_ip

def _get_evalgto_derivname(shortname: str, derivmode: str):
    if derivmode == "r":
        return "ip%s" % shortname
    else:
        raise RuntimeError("Unknown derivmode: %s" % derivmode)
