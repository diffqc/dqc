import os
import sys
import ctypes
import ctypes.util
import numpy as np
from typing import Dict, Any, Callable

# contains functions and constants that are used specifically for
# dqc.hamilton.intor files (no dependance on other files in dqc.hamilton.intor
# is required)

__all__ = ["NDIM", "CINT", "CGTO", "CPBC", "CSYMM", "c_null_ptr", "np2ctypes", "int2ctypes"]

# CONSTANTS
NDIM = 3

# libraries
_ext = "dylib" if sys.platform == "darwin" else "so"
_libcint_relpath = f"../../lib/libcint.{_ext}"
_libcgto_relpath = f"../../lib/libcgto.{_ext}"
_libcpbc_relpath = f"../../lib/libpbc.{_ext}"
# _libcvhf_relpath = f"../../lib/libcvhf.{_ext}"
_libcsymm_relpath = f"../../lib/libsymm.{_ext}"

_libs: Dict[str, Any] = {}

def _library_loader(name: str, relpath: str) -> Callable:
    curpath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(curpath, relpath))

    # load the library and cache the handler
    def fcn():
        if name not in _libs:
            try:
                _libs[name] = ctypes.cdll.LoadLibrary(path)
            except OSError as e:
                path2 = ctypes.util.find_library(name)
                if path2 is None:
                    raise e
                _libs[name] = ctypes.cdll.LoadLibrary(path2)
        return _libs[name]
    return fcn

CINT = _library_loader("cint", _libcint_relpath)
CGTO = _library_loader("cgto", _libcgto_relpath)
CPBC = _library_loader("cpbc", _libcpbc_relpath)
# CVHF = _library_loader("CVHF", _libcvhf_relpath)
CSYMM = _library_loader("symm", _libcsymm_relpath)

c_null_ptr = ctypes.POINTER(ctypes.c_void_p)

def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    # get the ctypes of the numpy ndarray
    return a.ctypes.data_as(ctypes.c_void_p)

def int2ctypes(a: int) -> ctypes.c_int:
    # convert the python's integer to ctypes' integer
    return ctypes.c_int(a)
