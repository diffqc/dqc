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

    def eval_gto(self, shortname: str, rgrid: torch.Tensor):
        # NOTE: this method do not propagate gradient and should only be used
        # in this file only

        # rgrid: (ngrid, ndim)

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
        ao_loc = np.asarray(self.cint.ao_loc, dtype=np.int32)

        c_shls = (ctypes.c_int * 2)(0, nshells)
        c_ngrid = ctypes.c_int(ngrid)
        # print("shells", (0, nshells))
        # print("ngrid", ngrid)
        # print("ao_loc", ao_loc)
        # print("ao_loc.dtype", ao_loc.dtype)
        # print("coords.shape", coords.shape)
        # print("coords.strides", coords.strides)
        # print("coords.dtype", coords.dtype)
        # print("out.shape", out.shape)
        # print("out.strides", out.strides)
        # print("non0tab.shape", non0tab.shape)
        # print("non0tab.strides", non0tab.strides)
        # print("_atm", self.cint._atm)
        # print("_bas", self.cint._bas)
        print("_env", self.cint._env)

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

if __name__ == "__main__":
    from ddft.basissets.cgtobasis import loadbasis
    dtype = torch.double
    pos1 = torch.tensor([0.0, 0.0,  0.8], dtype=dtype, requires_grad=True)
    pos2 = torch.tensor([0.0, 0.0, -0.8], dtype=dtype, requires_grad=True)
    n = 1000
    z = torch.linspace(-5, 5, n, dtype=dtype)
    zeros = torch.zeros(n, dtype=dtype)
    rgrid = torch.cat((zeros[None, :], zeros[None, :], z[None, :]), dim=0).T.contiguous().to(dtype)
    basis = "6-311++G**"

    def evalgto(pos1, pos2, rgrid, name):
        bases = loadbasis("1:%s" % basis, dtype=dtype, requires_grad=False)
        atombasis1 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos1)
        atombasis2 = AtomCGTOBasis(atomz=1, bases=bases, pos=pos2)
        env = LibcgtoWrapper([atombasis1, atombasis2], spherical=True)
        return env.eval_gto(name, rgrid)

    a = evalgto(pos1, pos2, rgrid, "")  # (nbasis, nr)

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
    plt.plot(z, a[i, :])
    plt.show()
