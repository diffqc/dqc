import torch
from ddft.csrc import _overlap, _kinetic, _nuclattr

def overlap(a1, pos1, lmn1, a2, pos2, lmn2):
    # a1: (*nx)
    # pos1: (3, *nx)
    # lmn1: (3, *nx)
    # returns: (*nx)

    # check shape
    a1shape = a1.shape
    pos1shape = pos1.shape
    assert a1shape == a2.shape and pos1shape[1:] == a1shape and \
           pos1shape == pos2.shape and pos1shape == lmn1.shape and \
           pos1shape == lmn2.shape

    return OverlapFunction.apply(
        a1.contiguous(),
        pos1.contiguous(),
        lmn1.to(torch.int).contiguous(),
        a2.contiguous(),
        pos2.contiguous(),
        lmn2.to(torch.int).contiguous(),
    )

class OverlapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, pos1, lmn1, a2, pos2, lmn2):
        x1, y1, z1 = pos1
        l1, m1, n1 = lmn1
        x2, y2, z2 = pos2
        l2, m2, n2 = lmn2
        res = _overlap(
            a1, x1, y1, z1, l1, m1, n1,
            a2, x2, y2, z2, l2, m2, n2)
        ctx.save_for_backward(a1, pos1, lmn1, a2, pos2, lmn2, res)
        return res

    @staticmethod
    def backward(ctx, grad_res):
        a1, pos1, lmn1, a2, pos2, lmn2, res = ctx.saved_tensors
        ndim = a1.ndim
        ix = torch.tensor([1, 0, 0], dtype=lmn1.dtype, device=lmn1.device)
        iy = torch.tensor([0, 1, 0], dtype=lmn1.dtype, device=lmn1.device)
        iz = torch.tensor([0, 0, 1], dtype=lmn1.dtype, device=lmn1.device)
        ix = ix[(..., ) + (None, ) * ndim]
        iy = iy[(..., ) + (None, ) * ndim]
        iz = iz[(..., ) + (None, ) * ndim]

        grad_a1 = None
        if a1.requires_grad:
            dsda1 = OverlapFunction.apply(a1, pos1, lmn1 + (ix * 2), a2, pos2, lmn2) + \
                    OverlapFunction.apply(a1, pos1, lmn1 + (iy * 2), a2, pos2, lmn2) + \
                    OverlapFunction.apply(a1, pos1, lmn1 + (iz * 2), a2, pos2, lmn2)
            grad_a1 = -dsda1 * grad_res

        grad_a2 = None
        if a2.requires_grad:
            dsda2 = OverlapFunction.apply(a1, pos1, lmn1, a2, pos2, lmn2 + (ix * 2)) + \
                    OverlapFunction.apply(a1, pos1, lmn1, a2, pos2, lmn2 + (iy * 2)) + \
                    OverlapFunction.apply(a1, pos1, lmn1, a2, pos2, lmn2 + (iz * 2))
            grad_a2 = -dsda2 * grad_res

        grad_pos1 = None
        grad_pos2 = None
        if pos1.requires_grad or pos2.requires_grad:
            dsdx1 = -lmn1[0] * OverlapFunction.apply(a1, pos1, lmn1 - ix, a2, pos2, lmn2) + \
                     2 * a1  * OverlapFunction.apply(a1, pos1, lmn1 + ix, a2, pos2, lmn2)
            dsdy1 = -lmn1[1] * OverlapFunction.apply(a1, pos1, lmn1 - iy, a2, pos2, lmn2) + \
                     2 * a1  * OverlapFunction.apply(a1, pos1, lmn1 + iy, a2, pos2, lmn2)
            dsdz1 = -lmn1[2] * OverlapFunction.apply(a1, pos1, lmn1 - iz, a2, pos2, lmn2) + \
                     2 * a1  * OverlapFunction.apply(a1, pos1, lmn1 + iz, a2, pos2, lmn2)

            dsdpos1 = torch.cat(
                (dsdx1.unsqueeze(0), dsdy1.unsqueeze(0), dsdz1.unsqueeze(0)), dim=0)
            grad_pos1 = dsdpos1 * grad_res
            grad_pos2 = -grad_pos1

        return grad_a1, grad_pos1, None, grad_a2, grad_pos2, None

if __name__ == "__main__":
    import time
    n = 2
    dtype = torch.double
    a1 = (torch.rand(n, dtype=dtype) + 0.1).requires_grad_()
    a2 = (torch.rand(n, dtype=dtype) + 0.1).requires_grad_()
    pos1 = (torch.randn((3, n), dtype=dtype) * 0.3).requires_grad_()
    pos2 = (torch.randn((3, n), dtype=dtype) * 0.3).requires_grad_()
    lmn1 = torch.ones((3, n))
    lmn2 = torch.ones((3, n))
    t0 = time.time()
    s = overlap(a1, pos1, lmn1, a2, pos2, lmn2)
    t1 = time.time()
    print(s)
    print(t1 - t0)
    torch.autograd.gradcheck(overlap, (a1, pos1, lmn1, a2, pos2, lmn2))
    torch.autograd.gradgradcheck(overlap, (a1, pos1, lmn1, a2, pos2, lmn2))
    t2 = time.time()
    print(t2 - t1)
