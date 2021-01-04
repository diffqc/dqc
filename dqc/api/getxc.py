import re
import pylibxc
from dqc.xc.base_xc import BaseXC
from dqc.xc.libxc import LibXCLDA, LibXCGGA

__all__ = ["get_libxc", "get_xc"]

def get_libxc(name: str) -> BaseXC:
    """
    Get the XC object of the libxc based on its libxc's name.

    Arguments
    ---------
    name: str
        The full libxc name, e.g. "lda_c_pw"

    Returns
    -------
    BaseXC
        XC object that wraps the xc requested
    """
    obj = pylibxc.LibXCFunctional(name, "unpolarized")
    family = obj.get_family()
    del obj
    if family == 1:  # LDA
        return LibXCLDA(name)
    elif family == 2:  # GGA
        return LibXCGGA(name)
    else:
        raise NotImplementedError("LibXC wrapper for family %d has not been implemented" % family)

def get_xc(xcstr: str) -> BaseXC:
    """
    Returns the XC object based on the expression in xcstr.

    Arguments
    ---------
    xcstr: str
        The expression of the xc string, e.g. "lda_x + gga_c_pbe" where the
        variable name will be replaced by the LibXC object

    Returns
    -------
    BaseXC
        XC object based on the given expression
    """
    # wrap the name of xc with "get_libxc"
    pattern = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"
    new_xcstr = re.sub(pattern, r'get_libxc("\1")', xcstr)

    # evaluate the expression and return the xc
    glob = {"get_libxc": get_libxc}
    return eval(new_xcstr, glob)
