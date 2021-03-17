import os
import ctypes
import numpy as np

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
