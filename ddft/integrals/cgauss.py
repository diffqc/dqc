import torch
from ddft.csrc import _overlap, _kinetic, _nuclattr, _elrep

__all__ = ["overlap", "kinetic", "nuclattr", "elrep"]

def overlap(a1, pos1, lmn1, a2, pos2, lmn2):
    return _apply_fcn(OverlapFunction.apply, (a1, a2), (pos1, pos2), (lmn1, lmn2))

def kinetic(a1, pos1, lmn1, a2, pos2, lmn2):
    return _apply_fcn(KineticFunction.apply, (a1, a2), (pos1, pos2), (lmn1, lmn2))

def nuclattr(a1, pos1, lmn1, a2, pos2, lmn2, posc):
    # making the atom at (0, 0, 0)
    pos1 = pos1 - posc
    pos2 = pos2 - posc
    return _apply_fcn(NuclattrFunction.apply, (a1, a2), (pos1, pos2), (lmn1, lmn2))

def elrep(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4):
    return _apply_fcn(ElrepFunction.apply, (a1, a2, a3, a4),
                      (pos1, pos2, pos3, pos4), (lmn1, lmn2, lmn3, lmn4))

# TODO: is there a more efficient way to calculate the backward instead of
# using 12 and 18 calls to itself?
# or maybe use cache to save previously calculated coefficients?

class OverlapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, pos1, lmn1, a2, pos2, lmn2):
        res = _calc_forward(_overlap, a1, pos1, lmn1, a2, pos2, lmn2)
        ctx.save_for_backward(a1, pos1, lmn1, a2, pos2, lmn2)
        return res

    @staticmethod
    def backward(ctx, grad_res):
        return _calc_backward(ctx.saved_tensors, grad_res, OverlapFunction.apply)

class KineticFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, pos1, lmn1, a2, pos2, lmn2):
        res = _calc_forward(_kinetic, a1, pos1, lmn1, a2, pos2, lmn2)
        ctx.save_for_backward(a1, pos1, lmn1, a2, pos2, lmn2)
        return res

    @staticmethod
    def backward(ctx, grad_res):
        return _calc_backward(ctx.saved_tensors, grad_res, KineticFunction.apply)

class NuclattrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, pos1, lmn1, a2, pos2, lmn2):
        res = _calc_forward(_nuclattr, a1, pos1, lmn1, a2, pos2, lmn2)
        ctx.save_for_backward(a1, pos1, lmn1, a2, pos2, lmn2)
        return res

    @staticmethod
    def backward(ctx, grad_res):
        return _calc_backward(ctx.saved_tensors, grad_res, NuclattrFunction.apply)

class ElrepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4):
        res = _calc_forward_el(_elrep, a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4)
        ctx.save_for_backward(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4)
        return res

    @staticmethod
    def backward(ctx, grad_res):
        return _calc_backward_el(ctx.saved_tensors, grad_res, ElrepFunction.apply)

def _apply_fcn(fcn, alphas, poss, lmns):
    n = len(alphas)
    assert n > 0
    assert all((a.shape == pos.shape[1:] == lmn.shape[1:]) \
           for (a, pos, lmn) in zip(alphas, poss, lmns))

    alphas = [a.contiguous().view(-1) for a in alphas]
    poss = [pos.contiguous().view(3, -1) for pos in poss]
    lmns = [lmn.to(torch.int).contiguous().view(3, -1) for lmn in lmns]
    inps = [(a, pos, lmn) for (a, pos, lmn) in zip(alphas, poss, lmns)]
    flat_inps = [item for sublist in inps for item in sublist]
    res = fcn(*flat_inps)  # (numel(a)^n)

    flat_shape = [item for sublist in ([alphas[0].shape] * n) for item in sublist]
    return res.view(*flat_shape)

def _calc_forward(cfunc, a1, pos1, lmn1, a2, pos2, lmn2):
    x1, y1, z1 = pos1
    l1, m1, n1 = lmn1
    x2, y2, z2 = pos2
    l2, m2, n2 = lmn2
    res = cfunc(
        a1, x1, y1, z1, l1, m1, n1,
        a2, x2, y2, z2, l2, m2, n2)
    return res

def _calc_backward(saved_tensors, grad_res, fcn):
    a1, pos1, lmn1, a2, pos2, lmn2 = saved_tensors
    ndim = a1.ndim
    ix = torch.tensor([1, 0, 0], dtype=lmn1.dtype, device=lmn1.device)
    iy = torch.tensor([0, 1, 0], dtype=lmn1.dtype, device=lmn1.device)
    iz = torch.tensor([0, 0, 1], dtype=lmn1.dtype, device=lmn1.device)
    ix = ix[(..., ) + (None, ) * ndim]
    iy = iy[(..., ) + (None, ) * ndim]
    iz = iz[(..., ) + (None, ) * ndim]

    grad_a1 = None
    if a1.requires_grad:
        dsda1 = fcn(a1, pos1, lmn1 + (ix * 2), a2, pos2, lmn2) + \
                fcn(a1, pos1, lmn1 + (iy * 2), a2, pos2, lmn2) + \
                fcn(a1, pos1, lmn1 + (iz * 2), a2, pos2, lmn2)
        grad_a1 = (-dsda1 * grad_res).sum(dim=-1)

    grad_a2 = None
    if a2.requires_grad:
        dsda2 = fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (ix * 2)) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (iy * 2)) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (iz * 2))
        grad_a2 = (-dsda2 * grad_res).sum(dim=-2)

    grad_pos1 = None
    if pos1.requires_grad:
        a1b = a1[:, None]  # (na, 1)
        lmn1b = lmn1[..., None]  # (3, na, 1)

        dsdx1 = -lmn1b[0] * fcn(a1, pos1, lmn1 - ix, a2, pos2, lmn2) + \
                 2 * a1b  * fcn(a1, pos1, lmn1 + ix, a2, pos2, lmn2)
        dsdy1 = -lmn1b[1] * fcn(a1, pos1, lmn1 - iy, a2, pos2, lmn2) + \
                 2 * a1b  * fcn(a1, pos1, lmn1 + iy, a2, pos2, lmn2)
        dsdz1 = -lmn1b[2] * fcn(a1, pos1, lmn1 - iz, a2, pos2, lmn2) + \
                 2 * a1b  * fcn(a1, pos1, lmn1 + iz, a2, pos2, lmn2)

        dsdpos1 = torch.cat(
            (dsdx1.unsqueeze(0), dsdy1.unsqueeze(0), dsdz1.unsqueeze(0)), dim=0)
        grad_pos1 = (dsdpos1 * grad_res).sum(dim=-1)

    grad_pos2 = None
    if pos2.requires_grad:
        dsdx2 = -lmn2[0] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - ix) + \
                 2 * a2  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + ix)
        dsdy2 = -lmn2[1] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - iy) + \
                 2 * a2  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + iy)
        dsdz2 = -lmn2[2] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - iz) + \
                 2 * a2  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + iz)

        dsdpos2 = torch.cat(
            (dsdx2.unsqueeze(0), dsdy2.unsqueeze(0), dsdz2.unsqueeze(0)), dim=0)
        grad_pos2 = (dsdpos2 * grad_res).sum(dim=-2)

    return grad_a1, grad_pos1, None, grad_a2, grad_pos2, None

def _calc_forward_el(cfunc, a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4):
    x1, y1, z1 = pos1
    l1, m1, n1 = lmn1
    x2, y2, z2 = pos2
    l2, m2, n2 = lmn2
    x3, y3, z3 = pos3
    l3, m3, n3 = lmn3
    x4, y4, z4 = pos4
    l4, m4, n4 = lmn4
    res = cfunc(
        a1, x1, y1, z1, l1, m1, n1,
        a2, x2, y2, z2, l2, m2, n2,
        a3, x3, y3, z3, l3, m3, n3,
        a4, x4, y4, z4, l4, m4, n4)
    return res

def _calc_backward_el(saved_tensors, grad_res, fcn):
    a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 = saved_tensors
    ndim = a1.ndim
    ix = torch.tensor([1, 0, 0], dtype=lmn1.dtype, device=lmn1.device)
    iy = torch.tensor([0, 1, 0], dtype=lmn1.dtype, device=lmn1.device)
    iz = torch.tensor([0, 0, 1], dtype=lmn1.dtype, device=lmn1.device)
    ix = ix[(..., ) + (None, ) * ndim]
    iy = iy[(..., ) + (None, ) * ndim]
    iz = iz[(..., ) + (None, ) * ndim]

    grad_a1 = None
    if a1.requires_grad:
        dsda1 = fcn(a1, pos1, lmn1 + (ix * 2), a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4) + \
                fcn(a1, pos1, lmn1 + (iy * 2), a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4) + \
                fcn(a1, pos1, lmn1 + (iz * 2), a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4)
        grad_a1 = (-dsda1 * grad_res).sum(dim=-1).sum(dim=-1).sum(dim=-1)

    grad_a2 = None
    if a2.requires_grad:
        dsda2 = fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (ix * 2), a3, pos3, lmn3, a4, pos4, lmn4) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (iy * 2), a3, pos3, lmn3, a4, pos4, lmn4) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (iz * 2), a3, pos3, lmn3, a4, pos4, lmn4)
        grad_a2 = (-dsda2 * grad_res).sum(dim=-1).sum(dim=-1).sum(dim=-2)

    grad_a3 = None
    if a3.requires_grad:
        dsda3 = fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 + (ix * 2), a4, pos4, lmn4) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 + (iy * 2), a4, pos4, lmn4) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 + (iz * 2), a4, pos4, lmn4)
        grad_a3 = (-dsda3 * grad_res).sum(dim=-1).sum(dim=-2).sum(dim=-2)

    grad_a4 = None
    if a4.requires_grad:
        dsda4 = fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 + (ix * 2)) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 + (iy * 2)) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 + (iz * 2))
        grad_a4 = (-dsda4 * grad_res).sum(dim=-2).sum(dim=-2).sum(dim=-2)

    grad_pos1 = None
    if pos1.requires_grad:
        lmn1b = lmn1[..., None, None, None]
        a1b = a1[..., None, None, None]
        dsdx1 = -lmn1b[0] * fcn(a1, pos1, lmn1 - ix, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4) + \
                 2 * a1b  * fcn(a1, pos1, lmn1 + ix, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4)
        dsdy1 = -lmn1b[1] * fcn(a1, pos1, lmn1 - iy, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4) + \
                 2 * a1b  * fcn(a1, pos1, lmn1 + iy, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4)
        dsdz1 = -lmn1b[2] * fcn(a1, pos1, lmn1 - iz, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4) + \
                 2 * a1b  * fcn(a1, pos1, lmn1 + iz, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4)

        dsdpos1 = torch.cat(
            (dsdx1.unsqueeze(0), dsdy1.unsqueeze(0), dsdz1.unsqueeze(0)), dim=0)
        grad_pos1 = (dsdpos1 * grad_res).sum(dim=-1).sum(dim=-1).sum(dim=-1)

    grad_pos2 = None
    if pos2.requires_grad:
        lmn2b = lmn2[..., None, None]
        a2b = a2[..., None, None]
        dsdx2 = -lmn2b[0] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - ix, a3, pos3, lmn3, a4, pos4, lmn4) + \
                 2 * a2b  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + ix, a3, pos3, lmn3, a4, pos4, lmn4)
        dsdy2 = -lmn2b[1] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - iy, a3, pos3, lmn3, a4, pos4, lmn4) + \
                 2 * a2b  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + iy, a3, pos3, lmn3, a4, pos4, lmn4)
        dsdz2 = -lmn2b[2] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - iz, a3, pos3, lmn3, a4, pos4, lmn4) + \
                 2 * a2b  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + iz, a3, pos3, lmn3, a4, pos4, lmn4)

        dsdpos2 = torch.cat(
            (dsdx2.unsqueeze(0), dsdy2.unsqueeze(0), dsdz2.unsqueeze(0)), dim=0)
        grad_pos2 = (dsdpos2 * grad_res).sum(dim=-1).sum(dim=-1).sum(dim=-2)

    grad_pos3 = None
    if pos3.requires_grad:
        lmn3b = lmn3[..., None]
        a3b = a3[..., None]
        dsdx3 = -lmn3b[0] * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 - ix, a4, pos4, lmn4) + \
                 2 * a3b  * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 + ix, a4, pos4, lmn4)
        dsdy3 = -lmn3b[1] * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 - iy, a4, pos4, lmn4) + \
                 2 * a3b  * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 + iy, a4, pos4, lmn4)
        dsdz3 = -lmn3b[2] * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 - iz, a4, pos4, lmn4) + \
                 2 * a3b  * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3 + iz, a4, pos4, lmn4)

        dsdpos3 = torch.cat(
            (dsdx3.unsqueeze(0), dsdy3.unsqueeze(0), dsdz3.unsqueeze(0)), dim=0)
        grad_pos3 = (dsdpos3 * grad_res).sum(dim=-1).sum(dim=-2).sum(dim=-2)

    grad_pos4 = None
    if pos4.requires_grad:
        dsdx4 = -lmn4[0] * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 - ix) + \
                 2 * a4  * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 + ix)
        dsdy4 = -lmn4[1] * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 - iy) + \
                 2 * a4  * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 + iy)
        dsdz4 = -lmn4[2] * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 - iz) + \
                 2 * a4  * fcn(a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4 + iz)

        dsdpos4 = torch.cat(
            (dsdx4.unsqueeze(0), dsdy4.unsqueeze(0), dsdz4.unsqueeze(0)), dim=0)
        grad_pos4 = (dsdpos4 * grad_res).sum(dim=-2).sum(dim=-2).sum(dim=-2)

    return grad_a1, grad_pos1, None, grad_a2, grad_pos2, None, grad_a3, grad_pos3, None, grad_a4, grad_pos4, None

if __name__ == "__main__":
    import time
    with_gradcheck = 1
    op = "elrep"
    n = 2 if with_gradcheck else 20000
    dtype = torch.double
    a1 = (torch.rand(n, dtype=dtype) + 0.1).requires_grad_()
    a2 = (torch.rand(n, dtype=dtype) + 0.1).requires_grad_()
    a3 = (torch.rand(n, dtype=dtype) + 0.1).requires_grad_()
    a4 = (torch.rand(n, dtype=dtype) + 0.1).requires_grad_()
    pos1 = (torch.randn((3, n), dtype=dtype) * 0.3).requires_grad_()
    pos2 = (torch.randn((3, n), dtype=dtype) * 0.3).requires_grad_()
    pos3 = (torch.randn((3, n), dtype=dtype) * 0.3).requires_grad_()
    pos4 = (torch.randn((3, n), dtype=dtype) * 0.3).requires_grad_()
    posc = (torch.randn((3, n), dtype=dtype) + 1)#.requires_grad_()
    lmn1 = torch.ones((3, n)) * 1
    lmn2 = torch.ones((3, n)) * 1
    lmn3 = torch.ones((3, n)) * 1
    lmn4 = torch.ones((3, n)) * 1
    params1 = (a1, pos1, lmn1, a2, pos2, lmn2)
    params2 = (a1, pos1, lmn1, a2, pos2, lmn2, posc)
    params4 = (a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4)
    t0 = time.time()
    if op == "overlap":
        s = overlap(*params1)
    elif op == "kinetic":
        s = kinetic(*params1)
    elif op == "nuclattr":
        s = nuclattr(*params2)
    elif op == "elrep":
        s = elrep(*params4)
    else:
        raise RuntimeError("Unknown op: %s" % op)
    t1 = time.time()
    print(s)
    print(t1 - t0)
    if with_gradcheck:
        if op == "overlap":
            torch.autograd.gradcheck(overlap, params1)
            torch.autograd.gradgradcheck(overlap, params1)
        elif op == "kinetic":
            torch.autograd.gradcheck(kinetic, params1)
            torch.autograd.gradgradcheck(kinetic, params1)
        elif op == "nuclattr":
            torch.autograd.gradcheck(nuclattr, params2)
            torch.autograd.gradgradcheck(nuclattr, params2)
        elif op == "elrep":
            torch.autograd.gradcheck(elrep, params4)
            torch.autograd.gradgradcheck(elrep, params4)
    t2 = time.time()
    print(t2 - t1)
