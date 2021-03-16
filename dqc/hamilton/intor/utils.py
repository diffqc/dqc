import os
import ctypes
import numpy as np

# CONSTANTS
NDIM = 3

# libraries
_curpath = os.path.dirname(os.path.abspath(__file__))
_libcint_path = os.path.join(_curpath, "../../../lib/libcint/build/libcint.so")
_libcgto_path = os.path.join(_curpath, "../../../lib/libcgto.so")
CINT = ctypes.cdll.LoadLibrary(_libcint_path)
CGTO = ctypes.cdll.LoadLibrary(_libcgto_path)

def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    # get the ctypes of the numpy ndarray
    return a.ctypes.data_as(ctypes.c_void_p)

def int2ctypes(a: int) -> ctypes.c_int:
    # convert the python's integer to ctypes' integer
    return ctypes.c_int(a)
