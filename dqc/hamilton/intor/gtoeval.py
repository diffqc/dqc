import re
import ctypes
from typing import Tuple, Optional
import torch
import numpy as np
from dqc.hamilton.intor.lcintwrap import LibcintWrapper
from dqc.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CGTO
from dqc.hamilton.intor.pbcintor import _get_default_kpts, _get_default_options, PBCIntOption
from dqc.utils.pbc import estimate_ovlp_rcut

__all__ = ["evl", "eval_gto", "eval_gradgto", "eval_laplgto",
           "pbc_evl", "pbc_eval_gto", "pbc_eval_gradgto", "pbc_eval_laplgto"]

BLKSIZE = 128  # same as lib/gto/grid_ao_drv.c

# evaluation of the gaussian basis
def evl(shortname: str, wrapper: LibcintWrapper, rgrid: torch.Tensor) -> torch.Tensor:
    # expand ao_to_atom to have shape of (nao, ndim)
    ao_to_atom = wrapper.ao_to_atom().unsqueeze(-1).expand(-1, NDIM)

    # rgrid: (ngrid, ndim)
    return _EvalGTO.apply(
        # tensors
        *wrapper.params, rgrid,

        # nontensors or int tensors
        ao_to_atom, wrapper, shortname)

def pbc_evl(shortname: str, wrapper: LibcintWrapper, rgrid: torch.Tensor,
            kpts: Optional[torch.Tensor] = None,
            options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # evaluate the basis in periodic boundary condition, i.e. evaluate
    # sum_L exp(i*k*L) * phi(r - L)
    # rgrid: (ngrid, ndim)
    # kpts: (nkpts, ndim)
    # ls: (nls, ndim)
    # returns: (*ncomp, nkpts, nao, ngrid)

    # get the default arguments
    kpts1 = _get_default_kpts(kpts, dtype=wrapper.dtype, device=wrapper.device)
    options1 = _get_default_options(options)

    # get the shifts
    coeffs, alphas, _ = wrapper.params
    rcut = estimate_ovlp_rcut(options1.precision, coeffs, alphas)
    assert wrapper.lattice is not None
    ls = wrapper.lattice.get_lattice_ls(rcut=rcut)  # (nls, ndim)

    # evaluate the gto
    exp_ikl = torch.exp(1j * torch.matmul(kpts1, ls.transpose(-2, -1)))  # (nkpts, nls)
    rgrid_shift = rgrid - ls.unsqueeze(-2)  # (nls, ngrid, ndim)
    ao = evl(shortname, wrapper, rgrid_shift.reshape(-1, NDIM))  # (*ncomp, nao, nls * ngrid)
    ao = ao.reshape(*ao.shape[:-1], ls.shape[0], -1)  # (*ncomp, nao, nls, ngrid)
    out = torch.einsum("kl,...alg->...kag", exp_ikl, ao.to(exp_ikl.dtype))  # (*ncomp, nkpts, nao, ngrid)
    return out

# shortcuts
def eval_gto(wrapper: LibcintWrapper, rgrid: torch.Tensor) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (nao, ngrid)
    return evl("", wrapper, rgrid)

def eval_gradgto(wrapper: LibcintWrapper, rgrid: torch.Tensor) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (ndim, nao, ngrid)
    return evl("ip", wrapper, rgrid)

def eval_laplgto(wrapper: LibcintWrapper, rgrid: torch.Tensor) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (nao, ngrid)
    return evl("lapl", wrapper, rgrid)

def pbc_eval_gto(wrapper: LibcintWrapper, rgrid: torch.Tensor,
                 kpts: Optional[torch.Tensor] = None,
                 options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # kpts: (nkpts, ndim)
    # return: (nkpts, nao, ngrid)
    return pbc_evl("", wrapper, rgrid, kpts, options)

def pbc_eval_gradgto(wrapper: LibcintWrapper, rgrid: torch.Tensor,
                     kpts: Optional[torch.Tensor] = None,
                     options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # kpts: (nkpts, ndim)
    # return: (nkpts, nao, ngrid)
    return pbc_evl("ip", wrapper, rgrid, kpts, options)

def pbc_eval_laplgto(wrapper: LibcintWrapper, rgrid: torch.Tensor,
                     kpts: Optional[torch.Tensor] = None,
                     options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # kpts: (nkpts, ndim)
    # return: (nkpts, nao, ngrid)
    return pbc_evl("lapl", wrapper, rgrid, kpts, options)

################## pytorch function ##################
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
