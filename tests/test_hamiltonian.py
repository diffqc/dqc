import torch
from ddft.utils.fd import finite_differences
from ddft.hamiltonian import Hamilton1P1D, HamiltonNP1D

def compare_grad_with_fd(fcn, args, idxs, eps=1e-6, rtol=1e-3, fd_to64=True, verbose=False):
    if not hasattr(eps, "__iter__"):
        eps = [eps for i in range(len(idxs))]
    if not hasattr(rtol, "__iter__"):
        rtol = [rtol for i in range(len(idxs))]

    # calculate the differentiable loss
    loss0 = fcn(*args)

    # zeroing the grad
    for idx in idxs:
        if args[idx].grad is not None:
            args[idx].grad.zero_()

    loss0.backward()
    grads = [args[idx].grad.data for idx in idxs]

    # compare with finite differences
    if fd_to64:
        argsfd = [arg.to(torch.float64) \
            if (type(arg) == torch.Tensor and arg.dtype == torch.float) \
            else arg \
            for arg in args]
    else:
        argsfd = args
    fds = [finite_differences(fcn, argsfd, idx, eps=eps[i]) for i,idx in enumerate(idxs)]

    for i in range(len(idxs)):
        ratio = grads[i] / fds[i]
        if verbose:
            print("Params #%d" % (idxs[i]))
            print("* grad:")
            print(grads[i])
            print("* fd:")
            print(fds[i])
            print("* ratio:")
            print(ratio)
        assert torch.allclose(ratio, torch.ones(1, dtype=ratio.dtype), rtol=rtol[i])

def test_hamilton1p1d_1():

    def getloss(rgrid, vext):
        wf, e = Hamilton1P1D.apply(rgrid.unsqueeze(0), vext.unsqueeze(0), 0)
        return (wf**4 + e**2).sum()

    rgrid = torch.linspace(-2, 2, 101).to(torch.float)
    vext = (rgrid * rgrid).requires_grad_()
    compare_grad_with_fd(getloss, (rgrid, vext), [1], eps=1e-4, rtol=5e-3)

    # setup with 64-bit precision
    rgrid64 = rgrid.to(torch.float64)
    vext64  = vext.to(torch.float64).detach().requires_grad_()
    compare_grad_with_fd(getloss, (rgrid64, vext64), [1], eps=1e-4, rtol=5e-5)

def test_hamiltonNp1d_1():

    def getloss2(rgrid, vext, iexc):
        wf, e = HamiltonNP1D.apply(rgrid.unsqueeze(0), vext.unsqueeze(0),
                                   iexc.unsqueeze(0))
        return (wf**4 + e**2).sum()

    rgrid = torch.linspace(-2, 2, 101).to(torch.float)
    vext = (rgrid * rgrid).requires_grad_()
    iexc = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).to(torch.long)
    compare_grad_with_fd(getloss2, (rgrid, vext, iexc), [1], eps=1e-4, rtol=5e-3)

    # setup with 64-bit precision
    rgrid64 = rgrid.to(torch.float64)
    vext64  = vext.to(torch.float64).detach().requires_grad_()
    compare_grad_with_fd(getloss2, (rgrid64, vext64, iexc), [1], eps=1e-4, rtol=5e-5)

if __name__ == "__main__":
    test_hamilton1p1d_1()
    test_hamiltonNp1d_1()
