import torch
from ddft.utils.misc import set_default_option
from ddft.maths.linesearch import line_search

def selfconsistent(f, x0, jinv0=1.0, **options):
    """
    Solve the root finder problem with Broyden's method.

    Arguments
    ---------
    * f: callable
        Callable that takes params as the input and output nfeat-outputs.
    * x0: torch.tensor (nbatch, nfeat)
        Initial value of parameters to be put in the function, f.
    * jinv0: float or torch.tensor (nbatch, nfeat, nfeat)
        The initial inverse of the Jacobian. If float, it will be the diagonal.
    * options: dict or None
        Options of the function.

    Returns
    -------
    * x: torch.tensor (nbatch, nfeat)
        The x that approximate f(x) = 0.
    """
    # set up the default options
    config = set_default_option({
        "max_niter": 20,
        "min_feps": 1e-6,
        "verbose": False,
    }, options)

    # pull out the options for fast access
    min_feps = config["min_feps"]
    verbose = config["verbose"]

    # pull out the parameters of x0
    nbatch, nfeat = x0.shape
    device = x0.device
    dtype = x0.dtype

    # set up the initial jinv
    jinv = _set_jinv0(jinv0, x0)

    # perform the Broyden iterations
    x = x0
    fx = f(x0) # (nbatch, nfeat)
    for i in range(config["max_niter"]):
        dxnew = -torch.bmm(jinv, fx.unsqueeze(-1)) # (nbatch, nfeat, 1)
        xnew = x + dxnew.squeeze(-1) # (nbatch, nfeat)
        fxnew = f(xnew)
        dfnew = fxnew - fx

        # update variables for the next iteration
        fx = fxnew
        x = xnew

        # check the stopping condition
        if verbose:
            print("Iter %3d: %.3e" % (i+1, fx.abs().max()))
        if torch.allclose(fx, torch.zeros_like(fx), atol=min_feps):
            break

    return x

def broyden(f, x0, jinv0=1.0, **options):
    """
    Solve the root finder problem with Broyden's method.

    Arguments
    ---------
    * f: callable
        Callable that takes params as the input and output nfeat-outputs.
    * x0: torch.tensor (nbatch, nfeat)
        Initial value of parameters to be put in the function, f.
    * jinv0: float or torch.tensor (nbatch, nfeat, nfeat)
        The initial inverse of the Jacobian. If float, it will be the diagonal.
    * options: dict or None
        Options of the function.

    Returns
    -------
    * x: torch.tensor (nbatch, nfeat)
        The x that approximate f(x) = 0.
    """
    # set up the default options
    config = set_default_option({
        "max_niter": 20,
        "min_feps": 1e-6,
        "verbose": False,
    }, options)

    # pull out the options for fast access
    min_feps = config["min_feps"]
    verbose = config["verbose"]

    # pull out the parameters of x0
    nbatch, nfeat = x0.shape
    device = x0.device
    dtype = x0.dtype

    # set up the initial jinv
    jinv = _set_jinv0(jinv0, x0)

    # perform the Broyden iterations
    x = x0
    fx = f(x0) # (nbatch, nfeat)
    for i in range(config["max_niter"]):
        dxnew = -torch.bmm(jinv, fx.unsqueeze(-1)) # (nbatch, nfeat, 1)
        xnew = x + dxnew.squeeze(-1) # (nbatch, nfeat)
        fxnew = f(xnew)
        dfnew = fxnew - fx

        # calculate the new jinv
        xtnew_jinv = torch.bmm(xnew.unsqueeze(1), jinv) # (nbatch, 1, nfeat)
        jinv_dfnew = torch.bmm(jinv, dfnew.unsqueeze(-1)) # (nbatch, nfeat, 1)
        xtnew_jinv_dfnew = torch.bmm(xtnew_jinv, dfnew.unsqueeze(-1)) # (nbatch, 1, 1)
        jinvnew = jinv + torch.bmm(dxnew - jinv_dfnew, xtnew_jinv) / xtnew_jinv_dfnew

        # update variables for the next iteration
        fx = fxnew
        jinv = jinvnew
        x = xnew

        # check the stopping condition
        if verbose:
            print("Iter %3d: %.3e" % (i+1, fx.abs().max()))
        if torch.allclose(fx, torch.zeros_like(fx), atol=min_feps):
            break

    return x

def lbfgs(f, x0, jinv0=1.0, **options):
    """
    Solve the root finder problem with L-BFGS method.

    Arguments
    ---------
    * f: callable
        Callable that takes params as the input and output nfeat-outputs.
    * x0: torch.tensor (nbatch, nfeat)
        Initial value of parameters to be put in the function, f.
    * jinv0: float or torch.tensor (nbatch, nfeat, nfeat)
        The initial inverse of the Jacobian. If float, it will be the diagonal.
    * options: dict or None
        Options of the function.

    Returns
    -------
    * x: torch.tensor (nbatch, nfeat)
        The x that approximate f(x) = 0.
    """
    config = set_default_option({
        "max_niter": 20,
        "min_feps": 1e-6,
        "max_memory": 10,
        "alpha0": 1.0,
        "linesearch": False,
        "verbose": False,
    }, options)

    # pull out the options for fast access
    min_feps = config["min_feps"]
    max_memory = config["max_memory"]
    verbose = config["verbose"]
    linesearch = config["linesearch"]
    alpha = config["alpha0"]

    # set up the initial jinv and the memories
    H0 = _set_jinv0_diag(jinv0, x0) # (nbatch, nfeat)
    sk_history = []
    yk_history = []
    rk_history = []

    def _apply_Vk(rk, sk, yk, grad):
        # sk: (nbatch, nfeat)
        # yk: (nbatch, nfeat)
        # rk: (nbatch, 1)
        return grad - (sk * grad).sum(dim=-1, keepdim=True) * rk * yk

    def _apply_VkT(rk, sk, yk, grad):
        # sk: (nbatch, nfeat)
        # yk: (nbatch, nfeat)
        # rk: (nbatch, 1)
        return grad - (yk * grad).sum(dim=-1, keepdim=True) * rk * sk

    def _apply_Hk(H0, sk_hist, yk_hist, rk_hist, gk):
        # H0: (nbatch, nfeat)
        # sk: (nbatch, nfeat)
        # yk: (nbatch, nfeat)
        # rk: (nbatch, 1)
        # gk: (nbatch, nfeat)
        nhist = len(sk_hist)
        if nhist == 0:
            return H0 * gk

        k = nhist - 1
        rk = rk_hist[k]
        sk = sk_hist[k]
        yk = yk_hist[k]

        # get the last term (rk * sk * sk.T)
        rksksk = (sk * gk).sum(dim=-1, keepdim=True) * rk * sk

        # calculate the V_(k-1)
        grad = gk
        grad = _apply_Vk(rk_hist[k], sk_hist[k], yk_hist[k], grad)
        grad = _apply_Hk(H0, sk_hist[:k], yk_hist[:k], rk_hist[:k], grad)
        grad = _apply_VkT(rk_hist[k], sk_hist[k], yk_hist[k], grad)
        return grad + rksksk

    def _line_search(xk, gk, dk, g):
        if linesearch:
            dx, dg, nit = line_search(dk, xk, gk, g)
            return xk + dx, gk + dg
        else:
            return xk + alpha*dk, g(xk + alpha*dk)

    # perform the main iteration
    xk = x0
    gk = f(xk)
    for k in range(config["max_niter"]):
        dk = -_apply_Hk(H0, sk_history, yk_history, rk_history, gk)
        xknew, gknew = _line_search(xk, gk, dk, f)

        # store the history
        sk = xknew - xk # (nbatch, nfeat)
        yk = gknew - gk
        inv_rhok = 1.0 / (sk * yk).sum(dim=-1, keepdim=True) # (nbatch, 1)
        sk_history.append(sk)
        yk_history.append(yk)
        rk_history.append(inv_rhok)
        if len(sk_history) > max_memory:
            sk_history = sk_history[-max_memory:]
            yk_history = yk_history[-max_memory:]
            rk_history = rk_history[-max_memory:]

        # update for the next iteration
        xk = xknew
        # alphakold = alphak
        gk = gknew

        # check the stopping condition
        if verbose:
            print("Iter %3d: %.3e" % (k+1, gk.abs().max()))
        if torch.allclose(gk, torch.zeros_like(gk), atol=min_feps):
            break

    return xk

def _set_jinv0(jinv0, x0):
    nbatch, nfeat = x0.shape
    dtype = x0.dtype
    device = x0.device
    if type(jinv0) == torch.Tensor:
        jinv = jinv0
    else:
        jinv = torch.eye(nfeat).unsqueeze(0).repeat(nbatch, 1, 1).to(dtype).to(device)
        jinv = jinv * jinv0
    return jinv

def _set_jinv0_diag(jinv0, x0):
    if type(jinv0) == torch.Tensor:
        jinv = jinv0
    else:
        jinv = torch.zeros_like(x0).to(x0.device) + jinv0
    return jinv

if __name__ == "__main__":

    dtype = torch.float
    A = torch.tensor([[[0.9703, 0.1178, 0.5345],
         [0.0629, 0.3352, 0.6431],
         [0.8756, 0.7564, 0.1121]]]).to(dtype)
    xtrue = torch.tensor([[0.8690, 0.4324, 0.9035]]).to(dtype)
    b = torch.bmm(A, xtrue.unsqueeze(-1)).squeeze(-1)

    def f(x):
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) - b

    x0 = torch.zeros_like(xtrue)
    jinv0 = 1.0
    x = lbfgs(f, x0, jinv0, verbose=True)
    # x = broyden(f, x0, jinv0, verbose=True)

    print(A)
    print(x)
    print(xtrue)
    print(f(x))
