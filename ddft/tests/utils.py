import torch
from ddft.utils.fd import finite_differences

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
