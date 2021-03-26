import os
import ctypes
import torch
import numpy as np
import scipy.special

# contains functions and constants that are used specifically for
# dqc.hamilton.intor files

__all__ = ["NDIM", "CINT", "CGTO", "CPBC", "c_null_ptr", "np2ctypes", "int2ctypes"]

# CONSTANTS
NDIM = 3

# libraries
_curpath = os.path.dirname(os.path.abspath(__file__))
_libcint_path = os.path.join(_curpath, "../../../lib/libcint/build/libcint.so")
_libcgto_path = os.path.join(_curpath, "../../../lib/libcgto.so")
_libcpbc_path = os.path.join(_curpath, "../../../lib/libpbc.so")
CINT = ctypes.cdll.LoadLibrary(_libcint_path)
CGTO = ctypes.cdll.LoadLibrary(_libcgto_path)
CPBC = ctypes.cdll.LoadLibrary(_libcpbc_path)

c_null_ptr = ctypes.POINTER(ctypes.c_void_p)

def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    # get the ctypes of the numpy ndarray
    return a.ctypes.data_as(ctypes.c_void_p)

def int2ctypes(a: int) -> ctypes.c_int:
    # convert the python's integer to ctypes' integer
    return ctypes.c_int(a)

def estimate_ovlp_rcut(precision: float, coeffs: torch.Tensor, alphas: torch.Tensor) -> float:
    # estimate the rcut for lattice sum to achieve the given precision
    # it is estimated based on the overlap integral
    langmom = 1
    C = (coeffs * coeffs + 1e-200) * (2 * langmom + 1) * alphas / precision
    r0 = torch.tensor(20.0, dtype=coeffs.dtype, device=coeffs.device)
    for i in range(2):
        r0 = torch.sqrt(2.0 * torch.log(C * (r0 * r0 * alphas) ** (langmom + 1) + 1.) / alphas)
    rcut = float(torch.max(r0).detach())
    return rcut

def estimate_ke_cutoff(precision: float, coeffs: torch.Tensor, alphas: torch.Tensor) -> float:
    # kinetic energy cut off estimation based on cubic lattice
    # based on _estimate_ke_cutoff from pyscf
    # https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/pbc/gto/cell.py#L498

    langmom = 1
    log_k0 = 3 + torch.log(alphas) / 2
    l2fac2 = scipy.special.factorial2(langmom * 2 + 1)
    a = precision * l2fac2 ** 2 * (4 * alphas) ** (langmom * 2 + 1) / (128 * np.pi ** 4 * coeffs ** 4)
    log_rest = torch.log(a)
    Ecut = 2 * alphas * (log_k0 * (4 * langmom + 3) - log_rest)
    Ecut[Ecut <= 0] = .5
    log_k0 = .5 * torch.log(Ecut * 2)
    Ecut = 2 * alphas * (log_k0 * (4 * langmom + 3) - log_rest)
    Ecut[Ecut <= 0] = .5
    Ecut_max = float(torch.max(Ecut).detach())

    return Ecut_max
