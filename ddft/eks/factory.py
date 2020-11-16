import pylibxc
from ddft.eks.libxc import LibXCLDA, LibXCGGA

__all__ = ["get_xc", "get_libxc"]

xlist = {
    "lda": "lda_x",
    "pbe": "gga_x_pbe",
}
clist = {
    "lda": "lda_c_pw",
    "pbe": "gga_c_pbe",
}

def _get_x(s):
    if s in xlist:
        s = xlist[s]
    return get_libxc(s)

def _get_c(s):
    if s in clist:
        s = clist[s]
    return get_libxc(s)

def get_libxc(name):
    obj = pylibxc.LibXCFunctional(name, "unpolarized")
    family = obj.get_family()
    del obj
    if family == 1:  # LDA
        return LibXCLDA(name)
    elif family == 2:  # GGA
        return LibXCGGA(name)
    else:
        raise NotImplementedError("LibXC wrapper for family %d has not been implemented" % family)

def get_xc(xcstr):
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
