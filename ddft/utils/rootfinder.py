import torch

def broyden(f, x0, jinv0=1.0, options=None):
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
    config = {
        "max_niter": 20,
        "min_feps": 1e-6,
    }
    if options is None:
        options = {}
    config.update(options)

    # pull out the options for fast access
    min_feps = config["min_feps"]

    # pull out the parameters of x0
    nbatch, nfeat = x0.shape
    device = x0.device
    dtype = x0.dtype

    # set up the initial jinv
    jinv = _set_jinv0(jinv0)

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
        print(fx.abs().max())
        if torch.allclose(fx, torch.zeros_like(fx), atol=min_feps):
            break

    return x

def lbfgs(f, x0, options=None):
    pass

def _set_jinv0(jinv0):
    if type(jinv0) == torch.Tensor:
        jinv = jinv0
    else:
        jinv = torch.eye(nfeat).unsqueeze(0).repeat(nbatch, 1, 1).to(dtype).to(device)
        jinv = jinv + jinv0
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
    x = broyden(f, x0, jinv0)

    print(A)
    print(x)
    print(xtrue)
    print(f(x))
