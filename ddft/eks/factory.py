from ddft.eks.xlda import xLDA
from ddft.eks.clda import cLDA_PW

__all__ = ["get_xc"]

def _get_x(xstr):
    return {
        "lda": xLDA,
    }[xstr]()

def _get_c(cstr):
    return {
        "lda": cLDA_PW,
        "lda_pw": cLDA_PW,
    }[cstr]()

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
