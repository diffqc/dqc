import torch
from ddft.csrc import _overlap, _kinetic, _nuclattr

__all__ = ["overlap", "kinetic", "nuclattr"]

def overlap(a1, pos1, lmn1, a2, pos2, lmn2):
    return _apply_fcn(OverlapFunction.apply, a1, pos1, lmn1, a2, pos2, lmn2)

def kinetic(a1, pos1, lmn1, a2, pos2, lmn2):
    return _apply_fcn(KineticFunction.apply, a1, pos1, lmn1, a2, pos2, lmn2)

def nuclattr(a1, pos1, lmn1, a2, pos2, lmn2, posc):
    # making the atom at (0, 0, 0)
    pos1 = pos1 - posc
    pos2 = pos2 - posc
    return _apply_fcn(NuclattrFunction.apply, a1, pos1, lmn1, a2, pos2, lmn2)

# TODO: is there a more efficient way to calculate the backward instead of
# using 12 and 18 calls to itself?

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
        return _calc_backward(ctx.saved_tensors, grad_res, NuclattrFunction.apply,
                              explicit_gradpos2=True)

def _apply_fcn(fcn, a1, pos1, lmn1, a2, pos2, lmn2):
    # a1: (*nx)
    # pos1: (3, *nx)
    # lmn1: (3, *nx)
    # returns: (*nx)

    # check shape (all must have the compliance shape, no broadcasting)
    a1shape = a1.shape
    pos1shape = pos1.shape
    assert a1shape == a2.shape and pos1shape[1:] == a1shape and \
           pos1shape == pos2.shape and pos1shape == lmn1.shape and \
           pos1shape == lmn2.shape

    return fcn(
        a1.contiguous(),
        pos1.contiguous(),
        lmn1.to(torch.int).contiguous(),
        a2.contiguous(),
        pos2.contiguous(),
        lmn2.to(torch.int).contiguous(),
    )

def _calc_forward(cfunc, a1, pos1, lmn1, a2, pos2, lmn2):
    x1, y1, z1 = pos1
    l1, m1, n1 = lmn1
    x2, y2, z2 = pos2
    l2, m2, n2 = lmn2
    res = cfunc(
        a1, x1, y1, z1, l1, m1, n1,
        a2, x2, y2, z2, l2, m2, n2)
    return res

def _calc_backward(saved_tensors, grad_res, fcn, explicit_gradpos2=False):
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
        grad_a1 = -dsda1 * grad_res

    grad_a2 = None
    if a2.requires_grad:
        dsda2 = fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (ix * 2)) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (iy * 2)) + \
                fcn(a1, pos1, lmn1, a2, pos2, lmn2 + (iz * 2))
        grad_a2 = -dsda2 * grad_res

    grad_pos1 = None
    if pos1.requires_grad:
        dsdx1 = -lmn1[0] * fcn(a1, pos1, lmn1 - ix, a2, pos2, lmn2) + \
                 2 * a1  * fcn(a1, pos1, lmn1 + ix, a2, pos2, lmn2)
        dsdy1 = -lmn1[1] * fcn(a1, pos1, lmn1 - iy, a2, pos2, lmn2) + \
                 2 * a1  * fcn(a1, pos1, lmn1 + iy, a2, pos2, lmn2)
        dsdz1 = -lmn1[2] * fcn(a1, pos1, lmn1 - iz, a2, pos2, lmn2) + \
                 2 * a1  * fcn(a1, pos1, lmn1 + iz, a2, pos2, lmn2)

        dsdpos1 = torch.cat(
            (dsdx1.unsqueeze(0), dsdy1.unsqueeze(0), dsdz1.unsqueeze(0)), dim=0)
        grad_pos1 = dsdpos1 * grad_res

    grad_pos2 = None
    if pos2.requires_grad:
        if explicit_gradpos2 or (grad_pos1 is None):
            dsdx2 = -lmn2[0] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - ix) + \
                     2 * a2  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + ix)
            dsdy2 = -lmn2[1] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - iy) + \
                     2 * a2  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + iy)
            dsdz2 = -lmn2[2] * fcn(a1, pos1, lmn1, a2, pos2, lmn2 - iz) + \
                     2 * a2  * fcn(a1, pos1, lmn1, a2, pos2, lmn2 + iz)

            dsdpos2 = torch.cat(
                (dsdx2.unsqueeze(0), dsdy2.unsqueeze(0), dsdz2.unsqueeze(0)), dim=0)
            grad_pos2 = dsdpos2 * grad_res
        else:
            grad_pos2 = -grad_pos1

    return grad_a1, grad_pos1, None, grad_a2, grad_pos2, None

if __name__ == "__main__":
    import time
    n = 20000
    dtype = torch.double
    a1 = (torch.rand(n, dtype=dtype) + 0.1).requires_grad_()
    a2 = (torch.rand(n, dtype=dtype) + 0.1).requires_grad_()
    pos1 = (torch.randn((3, n), dtype=dtype) * 0.3).requires_grad_()
    pos2 = (torch.randn((3, n), dtype=dtype) * 0.3).requires_grad_()
    posc = (torch.randn((3, n), dtype=dtype) + 1)#.requires_grad_()
    lmn1 = torch.ones((3, n))
    lmn2 = torch.ones((3, n))
    t0 = time.time()
    # s = overlap(a1, pos1, lmn1, a2, pos2, lmn2)
    # s = kinetic(a1, pos1, lmn1, a2, pos2, lmn2)
    s = nuclattr(a1, pos1, lmn1, a2, pos2, lmn2, posc)
    t1 = time.time()
    print(s)
    print(t1 - t0)
    # torch.autograd.gradcheck(overlap, (a1, pos1, lmn1, a2, pos2, lmn2))
    # torch.autograd.gradgradcheck(overlap, (a1, pos1, lmn1, a2, pos2, lmn2))
    # torch.autograd.gradcheck(kinetic, (a1, pos1, lmn1, a2, pos2, lmn2))
    # torch.autograd.gradgradcheck(kinetic, (a1, pos1, lmn1, a2, pos2, lmn2))
    # torch.autograd.gradcheck(nuclattr, (a1, pos1, lmn1, a2, pos2, lmn2, posc))
    # torch.autograd.gradgradcheck(nuclattr, (a1, pos1, lmn1, a2, pos2, lmn2, posc))
    t2 = time.time()
    print(t2 - t1)
