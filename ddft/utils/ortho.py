import torch

def orthogonalize(V, r):
    # V: (nbatch, na, ncols)
    # r: (nbatch, na, 1)

    Vdotr = (V * r).sum(dim=-2, keepdim=True) # (nbatch, 1, ncols)
    Vproj = (V * Vdotr).sum(dim=-1, keepdim=True) # (nbatch, na, 1)
    rortho = r - Vproj
    rortho = rortho / rortho.norm(dim=-2, keepdim=True)
    return rortho

def biorthogonalize(V1, V2, r):
    V2dotr = (V2 * r).sum(dim=-2, keepdim=True)
    Vproj = (V1 * V2dotr).sum(dim=-1, keepdim=True)
    rbiortho = r - Vproj # (nbatch, na, 1)
    rbiortho = rbiortho / rbiortho.norm(dim=-2, keepdim=True)
    return rbiortho
