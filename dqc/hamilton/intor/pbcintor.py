from typing import Optional, List, Tuple, Callable
import ctypes
import copy
import re
import operator
import warnings
from dataclasses import dataclass
from functools import reduce
import numpy as np
import torch
from dqc.hamilton.intor.lcintwrap import LibcintWrapper
from dqc.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CINT, CPBC, \
                                     CGTO, c_null_ptr
from dqc.hamilton.intor.molintor import _check_and_set, Intor, _get_intgl_name, \
                                        _get_intgl_components_shape
from dqc.utils.types import get_complex_dtype

__all__ = ["PBCIntOption", "pbc_int1e",
           "pbc_overlap", "pbc_kinetic", "pbc_nuclattr"]

@dataclass
class PBCIntOption:
    """
    PBCIntOption is a class that contains parameters for the PBC integrals.
    """
    # TODO: any other options (?)
    precision: float = 1e-8

def pbc_int1e(shortname: str, wrapper: LibcintWrapper,
              other: Optional[LibcintWrapper] = None,
              kpts: Optional[torch.Tensor] = None,
              options: Optional[PBCIntOption] = None,
              ):
    # performing the periodic boundary condition (PBC) integrals on 1-electron

    # check and set the default values
    other1 = _check_and_set_pbc(wrapper, other)
    if options is None:
        options1 = PBCIntOption()
    else:
        options1 = options
    if kpts is None:
        kpts1 = torch.zeros((1, 3), dtype=wrapper.dtype, device=wrapper.device)
    else:
        kpts1 = kpts

    return _PBCInt2cFunction.apply(
        *wrapper.params,
        *wrapper.lattice.params,
        kpts1,
        [wrapper, other1],
        "int1e", shortname, options1)

# shortcuts
def pbc_overlap(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None,
                kpts: Optional[torch.Tensor] = None,
                options: Optional[PBCIntOption] = None) -> torch.Tensor:
    return pbc_int1e("ovlp", wrapper, other=other, kpts=kpts, options=options)

def pbc_kinetic(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None,
            kpts: Optional[torch.Tensor] = None,
            options: Optional[PBCIntOption] = None) -> torch.Tensor:
    return pbc_int1e("kin", wrapper, other=other, kpts=kpts, options=options)

def pbc_nuclattr(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None,
             kpts: Optional[torch.Tensor] = None,
             options: Optional[PBCIntOption] = None) -> torch.Tensor:
    return int1e("nuc", wrapper, other=other, kpts=kpts, options=options)

################# torch autograd function wrappers #################
class _PBCInt2cFunction(torch.autograd.Function):
    # wrapper class for the periodic boundary condition 2-centre integrals
    @staticmethod
    def forward(ctx,  # type: ignore
                # basis params
                allcoeffs: torch.Tensor, allalphas: torch.Tensor, allposs: torch.Tensor,
                # lattice params
                alattice: torch.Tensor,
                # other parameters
                kpts: torch.Tensor,
                wrappers: List[LibcintWrapper], int_type: str, shortname: str,
                options: PBCIntOption) -> torch.Tensor:
        # allcoeffs: (ngauss_tot,)
        # allalphas: (ngauss_tot,)
        # allposs: (natom, ndim)

        out_tensor = PBCIntor(int_type, shortname, wrappers, kpts, options).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs, alattice, kpts)
        ctx.other_info = (wrappers, int_type, shortname, options)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        pass

################# integrator object (direct interface to lib*) #################
class PBCIntor(object):
    def __init__(self, int_type: str, shortname: str, wrappers: List[LibcintWrapper],
                 kpts: torch.Tensor, options: PBCIntOption):
        # This is a class for once integration only
        # I made a class for refactoring reason because the integrals share
        # some parameters
        # No gradients propagated in the methods of this class

        assert len(wrappers) > 0
        wrapper0 = wrappers[0]
        kpts_np = kpts.detach().numpy()  # (nk, ndim)
        opname = _get_intgl_name(int_type, shortname, wrapper0.spherical)
        lattice = wrapper0.lattice
        self.int_type = int_type

        # 2-centre integral (TODO: move it to int2c once int4c or int3c are known)
        assert len(wrappers) == 2
        # libpbc will do in-place shift of the basis of one of the wrappers, so
        # we need to make a concatenated copy of the wrapper's atm_bas_env
        atm, bas, env, ao_loc = _concat_atm_bas_env(wrappers[0], wrappers[1])
        i0, i1 = wrappers[0].shell_idxs
        j0, j1 = wrappers[1].shell_idxs
        nshls0 = len(wrappers[0])
        shls_slice = (i0, i1, j0 + nshls0, j1 + nshls0)

        # prepare the output
        nkpts = len(kpts_np)
        comp_shape = _get_intgl_components_shape(shortname)
        ncomp = reduce(operator.mul, comp_shape, 1)
        outshape = (nkpts,) + comp_shape + tuple(w.nao() for w in wrappers)
        out = np.empty(outshape, dtype=np.complex128)

        # TODO: add symmetry here
        fill = CPBC.PBCnr2c_fill_ks1
        fintor = getattr(CGTO, opname)
        # TODO: use proper optimizers
        cintopt = c_null_ptr()
        cpbcopt = c_null_ptr()

        # estimate the rcut
        coeffs, alphas, _ = wrapper0.params
        l = 1
        C = (coeffs * coeffs + 1e-200) * (2 * l + 1) * alphas / options.precision
        r0 = 20.0
        for i in range(2):
            r0 = torch.sqrt(2.0 * torch.log(C * (r0 * r0 * alphas) ** (l + 1) + 1.) / alphas)
        rcut = torch.max(r0)

        # get the lattice translation vectors and the exponential factors
        ls = np.asarray(lattice.get_lattice_ls(rcut=rcut))
        expkl = np.asarray(np.exp(1j * np.dot(kpts_np, ls.T)), order='C')

        # if the ls is too big, it might produce segfault
        if (ls.shape[0] > 1e6):
            warnings.warn("The number of neighbors in the integral is too many, "\
                          "it might causes segfault")

        # perform the integration
        drv = CPBC.PBCnr2c_drv
        drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
            int2ctypes(nkpts), int2ctypes(ncomp), int2ctypes(len(ls)),
            np2ctypes(ls),
            np2ctypes(expkl),
            (ctypes.c_int * len(shls_slice))(*shls_slice),
            np2ctypes(ao_loc),
            cintopt, cpbcopt,
            np2ctypes(atm), int2ctypes(atm.shape[0]),
            np2ctypes(bas), int2ctypes(bas.shape[0]),
            np2ctypes(env), int2ctypes(env.size))

        self.out = torch.as_tensor(out, dtype=get_complex_dtype(wrapper0.dtype),
                                   device=wrapper0.device)
        # this class is meant to be used once
        self.integral_done = False

    def calc(self) -> torch.Tensor:
        assert not self.integral_done
        self.integral_done = True
        if self.int_type == "int1e":
            return self._int2c()
        else:
            raise ValueError("Unknown integral type: %s" % self.int_type)

    def _int2c(self) -> torch.Tensor:
        # this function works mostly in numpy
        # no gradients propagated in this function (and it's OK)
        return self.out

################# helper functions #################
def _check_and_set_pbc(wrapper: LibcintWrapper, other: Optional[LibcintWrapper]) -> LibcintWrapper:
    # check the `other` parameter if it is compatible to `wrapper`, then return
    # the `other` parameter (set to wrapper if it is `None`)
    other1 = _check_and_set(wrapper, other)
    assert other1.lattice is wrapper.lattice
    return other1

def _concat_atm_bas_env(wrapper: LibcintWrapper, other: LibcintWrapper) -> Tuple[np.ndarray, ...]:
    # make a copy of the concatenated atm, bas, env, and also return the new
    # ao_location

    # code from pyscf:
    # https://github.com/pyscf/pyscf/blob/e6c569932d5bab5e49994ae3dd365998fc5202b5/pyscf/gto/mole.py#L629
    atm1, bas1, env1 = wrapper.atm_bas_env
    atm2, bas2, env2 = other.atm_bas_env

    PTR_COORD = 1
    PTR_ZETA = 3
    ATOM_OF = 0
    PTR_EXP = 5
    PTR_COEFF = 6

    off = len(env1)
    natm_off = len(atm1)
    atm2 = np.copy(atm2)
    bas2 = np.copy(bas2)
    atm2[:, PTR_COORD] += off
    atm2[:, PTR_ZETA ] += off
    bas2[:, ATOM_OF  ] += natm_off
    bas2[:, PTR_EXP  ] += off
    bas2[:, PTR_COEFF] += off

    # get the new ao_loc
    ao_loc1 = wrapper.full_shell_to_aoloc
    ao_loc2 = other.full_shell_to_aoloc + ao_loc1[-1]
    ao_loc = np.concatenate((ao_loc1[:-1], ao_loc2))

    return (np.asarray(np.vstack((atm1, atm2)), dtype=np.int32),
            np.asarray(np.vstack((bas1, bas2)), dtype=np.int32),
            np.hstack((env1, env2)),
            ao_loc,
            )
