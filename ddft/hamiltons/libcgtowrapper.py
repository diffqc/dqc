import os
import re
import ctypes
from typing import List
import torch
import numpy as np
from ddft.basissets.cgtobasis import AtomCGTOBasis
from ddft.hamiltons.libcintwrapper import LibcintWrapper

NDIM = 3
BLKSIZE = 128  # same as lib/gto/grid_ao_drv.c

# load the libcint
_curpath = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_curpath, "../../lib/libcgto.so")
cgto = ctypes.cdll.LoadLibrary(_lib_path)

class LibcgtoWrapper(object):
    # this class provides the evaluation of contracted gaussian orbitals
    def __init__(self, atombases: List[AtomCGTOBasis], spherical: bool = True) -> None:
        self.cint = LibcintWrapper(atombases, spherical)
        self.spherical = spherical

    def _get_name(self, shortname: str) -> str:
        sname = ("_" + shortname) if (shortname != "") else ""
        suffix = "_sph" if self.spherical else "_cart"
        return "GTOval%s%s" % (sname, suffix)

    def _get_outshape(self, shortname: str, nao: int, ngrid: int) -> List[int]:
        # count "ip" only at the beginning
        n_ip = len(re.findall(r"^(?:ip)*(?:ip)?", shortname)[0]) // 2
        return ([NDIM] * n_ip) + [nao, ngrid]

    def _get_deriv_name(self, shortname: str, derivmode: str):
        if derivmode == "r":
            return "ip%s" % shortname
        else:
            raise RuntimeError("Unknown derivmode: %s" % derivmode)

    def evalgto(self, shortname: str, rgrid: torch.Tensor) -> torch.Tensor:
        # expand ao_to_atom to have shape of (nao, ndim)
        ao_to_atom = self.cint.ao_to_atom.unsqueeze(-1).expand(-1, NDIM)

        # rgrid: (ngrid, ndim)
        return _EvalGTO.apply(
            # tensors
            self.cint.allalphas_params,
            self.cint.allcoeffs_params,
            self.cint.allpos_params,
            rgrid,

            # nontensors or int tensors
            ao_to_atom,
            self,
            shortname)

    def eval_gto_internal(self, shortname: str, rgrid: torch.Tensor) -> torch.Tensor:
        # NOTE: this method do not propagate gradient and should only be used
        # in this file only

        # rgrid: (ngrid, ndim)
        # returns: (*, nao, ngrid)

        ngrid = rgrid.shape[0]
        nshells = self.cint.nshells_tot
        nao = self.cint.nao_tot
        opname = self._get_name(shortname)
        outshape = self._get_outshape(shortname, nao, ngrid)

        out = np.empty(outshape, dtype=np.float64)
        non0tab = np.ones(((ngrid + BLKSIZE - 1) // BLKSIZE, nshells),
                          dtype=np.int8)

        # TODO: check if we need to transpose it first?
        rgrid = rgrid.contiguous()
        coords = np.asarray(rgrid, dtype=np.float64, order='F')
        ao_loc = np.asarray(self.cint.shell_to_aoloc, dtype=np.int32)

        c_shls = (ctypes.c_int * 2)(0, nshells)
        c_ngrid = ctypes.c_int(ngrid)

        # evaluate the orbital
        operator = getattr(cgto, opname)
        operator.restype = ctypes.c_double
        operator(c_ngrid, c_shls,
                 ao_loc.ctypes.data_as(ctypes.c_void_p),
                 out.ctypes.data_as(ctypes.c_void_p),
                 coords.ctypes.data_as(ctypes.c_void_p),
                 non0tab.ctypes.data_as(ctypes.c_void_p),
                 self.cint.atm_ctypes, self.cint.natm_ctypes,
                 self.cint.bas_ctypes, self.cint.nbas_ctypes,
                 self.cint.env_ctypes)

        out = torch.tensor(out, dtype=self.cint.dtype, device=self.cint.device)
        return out

class _EvalGTO(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                # tensors not used in calculating the forward, but required
                # for the backward propagation
                alphas: torch.Tensor,  # (ngauss_tot)
                coeffs: torch.Tensor,  # (ngauss_tot)
                pos: torch.Tensor,  # (natom, ndim)

                # tensors used in forward
                rgrid: torch.Tensor,  # (ngrid, ndim)

                # other non-tensor info
                ao_to_atom: torch.Tensor,  # int tensor (nao, ndim)
                wrapper: LibcgtoWrapper,
                shortname: str) -> torch.Tensor:

        res = wrapper.eval_gto_internal(shortname, rgrid)  # (*, nao, ngrid)
        ctx.save_for_backward(alphas, coeffs, pos, rgrid)
        ctx.other_info = (ao_to_atom, wrapper, shortname)
        return res

    @staticmethod
    def backward(ctx, grad_res):
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
            opsname = wrapper._get_deriv_name(shortname, "r")
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

if __name__ == "__main__":
    from ddft.basissets.cgtobasis import loadbasis
    dtype = torch.double
    pos1 = torch.tensor([0.0, 0.0,  0.8], dtype=dtype, requires_grad=True)
    pos2 = torch.tensor([0.0, 0.0, -0.8], dtype=dtype, requires_grad=True)
    gradcheck = True
    n = 3 if gradcheck else 1000
    z = torch.linspace(-5, 5, n, dtype=dtype)
    zeros = torch.zeros(n, dtype=dtype)
    rgrid = torch.cat((zeros[None, :], zeros[None, :], z[None, :]), dim=0).T.contiguous().to(dtype)
    # basis = "6-311++G**"
    basis = "3-21G"

    def evalgto(pos1, pos2, rgrid, name):
        bases = loadbasis("1:%s" % basis, dtype=dtype, requires_grad=False)
        atombasis1 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos2)
        env = LibcgtoWrapper([atombasis1, atombasis2], spherical=True)
        return env.evalgto(name, rgrid)

    a = evalgto(pos1, pos2, rgrid, "")  # (nbasis, nr)
    if gradcheck:
        torch.autograd.gradcheck(evalgto, (pos1, pos2, rgrid, ""))
        torch.autograd.gradgradcheck(evalgto, (pos1, pos2, rgrid, ""))
    else:
        import pyscf
        import numpy as np

        mol = pyscf.gto.M(atom="H 0 0 0.8; H 0 0 -0.8", basis=basis, unit="Bohr")
        coords = np.zeros((n, 3))
        coords[:, 2] = np.linspace(-5, 5, n)
        ao_value = mol.eval_gto("GTOval_cart", coords)
        print(np.abs(ao_value).sum(axis=0))

        import matplotlib.pyplot as plt
        i = 0
        plt.plot(z, ao_value[:, i])
        plt.plot(z, a[i, :].detach())
        plt.show()
