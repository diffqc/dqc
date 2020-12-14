import pylibxc
from dqc.xc.base_xc import BaseXC
from dqc.xc.libxc import LibXCLDA, LibXCGGA

__all__ = ["get_libxc", "get_xc"]

xlist = {
    "lda": "lda_x",
    "pbe": "gga_x_pbe",
}
clist = {
    "lda": "lda_c_pw",
    "pbe": "gga_c_pbe",
}

def _get_x(s: str) -> BaseXC:
    if s in xlist:
        s = xlist[s]
    return get_libxc(s)

def _get_c(s: str) -> BaseXC:
    if s in clist:
        s = clist[s]
    return get_libxc(s)

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
    xclist = [s.strip().lower() for s in xcstr.split(",")]
    if len(xclist) == 1:
        s = xclist[0]
        return _get_x(s) + _get_c(s)
    elif len(xclist) == 2:
        xempty = xclist[0] == ""
        cempty = xclist[1] == ""
        if xempty and not cempty:
            return _get_c(xclist[1])
        elif not xempty and cempty:
            return _get_x(xclist[0])
        elif not xempty and not cempty:
            return _get_x(xclist[0]) + _get_c(xclist[1])
        else:
            raise ValueError("Invalid xc string: '%s'" % xcstr)
    else:
        raise ValueError("Invalid xc string: '%s'" % xcstr)
