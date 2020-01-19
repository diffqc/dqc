import torch
from ddft.utils.misc import set_default_option

"""
This file contains methods to obtain eigenpairs of a linear transformation
    which is a subclass of ddft.modules.base_linear.BaseLinearModule
"""

def lanczos(A, neig, params, **options):
    """
    Lanczos iterative method to obtain the `neig` lowest eigenpairs.
    This function is written so that the backpropagation can be done.

    Arguments
    ---------
    * A: BaseLinearModule instance
        The linear module object on which the eigenpairs are constructed.
    * neig: int
        The number of eigenpairs to be retrieved.
    * params: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to A.forward(x,*params).
        Each of params must have shape of (nbatch,...)
    * **options:
        Iterative algorithm options.

    Returns
    -------
    * eigvals: torch.tensor (nbatch, neig)
    * eigvecs: torch.tensor (nbatch, na, neig)
        The `neig` smallest eigenpairs
    """
    config = set_default_option({
        "max_niter": 120,
        "min_eps": 1e-6,
        "verbose": False,
        "v_init": "randn",
    }, options)
    verbose = config["verbose"]
    min_eps = config["min_eps"]

    # get the shape of the transformation
    na = _check_and_get_shape(A)
    nbatch = params[0].shape[0]
    dtype, device = _get_dtype_device(params, A)

    # set up the initial Krylov space
    V = _set_initial_v(config["v_init"].lower(), nbatch, na, 1) # (nbatch,na,1)
    V = V.to(dtype).to(device)

    prev_eigvals = None
    stop_reason = "max_niter"
    for i in range(config["max_niter"]):
        v = V[:,:,i] # (nbatch, na)
        Av = A(v, *params) # (nbatch, na)

        # construct the Krylov space
        V = torch.cat((V, Av.unsqueeze(-1)), dim=-1) # (nbatch, na, i+1)

        # orthogonalize
        Q, R = torch.qr(V) # Q: (nbatch, na, i+1)
        V = Q

        AV = A(V, *params) # (nbatch, na, i+1)
        T = torch.bmm(V.transpose(-2,-1), AV) # (nbatch, i+1, i+1)

        # check convergence
        if i+1 < neig: continue
        eigvalT, eigvecT = torch.symeig(T, eigenvectors=True) # val: (nbatch, i+1)
        eigvecA = torch.bmm(V, eigvecT) # (nbatch, na, i+1)
        if prev_eigvals is not None:
            dev = (eigvalT[:,:neig].data - prev_eigvals).abs().max()
            if verbose:
                print("Iter %3d (guess size: %d): %.3e" % (i, eigvecA.shape[-1], dev))
            if dev < min_eps:
                stop_reason = "min_eps"
                break
        prev_eigvals = eigvalT[:,:neig]

    eigvals = eigvalT[:,:neig]
    eigvecs = eigvecA[:,:,:neig]
    return eigvals, eigvecs

def davidson(A, neig, params, **options):
    """
    Iterative methods to obtain the `neig` lowest eigenvalues and eigenvectors.
    This function is written so that the backpropagation can be done.

    Arguments
    ---------
    * A: BaseLinearModule instance
        The linear module object on which the eigenpairs are constructed.
    * neig: int
        The number of eigenpairs to be retrieved.
    * params: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to A.forward(x,*params).
        Each of params must have shape of (nbatch,...)
    * **options:
        Iterative algorithm options.

    Returns
    -------
    * eigvals: torch.tensor (nbatch, neig)
    * eigvecs: torch.tensor (nbatch, na, neig)
        The `neig` smallest eigenpairs
    """
    config = set_default_option({
        "max_niter": 120,
        "nguess": neig+1,
        "min_eps": 1e-6,
        "verbose": False,
        "eps_cond": 1e-6,
        "v_init": "randn",
        "max_addition": 9e99,
    }, options)

    # get some of the options
    nguess = config["nguess"]
    max_niter = config["max_niter"]
    min_eps = config["min_eps"]
    verbose = config["verbose"]
    eps_cond = config["eps_cond"]
    max_addition = config["max_addition"]

    # get the shape of the transformation
    na = _check_and_get_shape(A)
    nbatch = params[0].shape[0]
    dtype, device = _get_dtype_device(params, A)

    # set up the initial guess
    V = _set_initial_v(config["v_init"].lower(), nbatch, na, nguess) # (nbatch,na,nguess)
    V = V.to(dtype).to(device)
    dA = A.diag(*params) # (nbatch, na)

    prev_eigvals = None
    stop_reason = "max_niter"
    for m in range(nguess, max_niter, nguess):
        VT = V.transpose(-2, -1)
        # Can be optimized by saving AV from the previous iteration and only
        # operate AV for the new V. This works because the old V has already
        # been orthogonalized, so it will stay the same
        AV = A(V, *params) # (nbatch, na, nguess)
        T = torch.bmm(VT, AV) # (nbatch, nguess, nguess)

        # eigvals are sorted from the lowest
        # eval: (nbatch, nguess), evec: (nbatch, nguess, nguess)
        eigvalT, eigvecT = torch.symeig(T, eigenvectors=True)
        eigvecA = torch.bmm(V, eigvecT) # (nbatch, na, nguess)

        # check the convergence
        if prev_eigvals is not None:
            dev = (eigvalT[:,:neig].data - prev_eigvals).abs().max()
            if verbose:
                print("Iter %3d (guess size: %d): %.3e" % (m, eigvecA.shape[-1], dev))
            if dev < min_eps:
                stop_reason = "min_eps"
                break

        # stop if V has become a full-rank matrix
        if V.shape[-1] == na:
            stop_reason = "full_rank"
            break

        # calculate the parameters for the next iteration
        prev_eigvals = eigvalT[:,:neig].data

        nj = eigvalT.shape[1]
        ritz_list = []
        nadd = min(nj, max_addition)
        for i in range(nadd):
            # precondition the inverse diagonal
            finv = dA - eigvalT[:,i:i+1]
            finv[finv.abs() < eps_cond] = eps_cond
            f = 1. / finv # (nbatch, na)

            AVphi = A(eigvecA[:,:,i], *params) # (nbatch, na)
            lmbdaVphi = eigvalT[:,i:i+1] * eigvecA[:,:,i] # (nbatch, na)
            r = f * (AVphi - lmbdaVphi) # (nbatch, na)
            ritz_list.append(r.unsqueeze(-1))

        # add the ritz vectors to the guess vectors
        ritz = torch.cat(ritz_list, dim=-1)
        ritz = ritz / ritz.norm(dim=1, keepdim=True) # (nbatch, na, nguess)
        Va = torch.cat((V, ritz), dim=-1)
        if Va.shape[-1] > na:
            Va = Va[:,:,-na:]

        # R^{-T} is needed for the backpropagation, so small det(R) will cause
        # numerical instability. So we need to choose ritz that are
        # perpendicular to the current columns of V

        # orthogonalize the new columns of V
        Q, R = torch.qr(Va) # V: (nbatch, na, nguess), R: (nbatch, nguess, nguess)
        V = Q

    eigvals = eigvalT[:,:neig]
    eigvecs = eigvecA[:,:,:neig]
    return eigvals, eigvecs

def exacteig(A, neig, params, **options):
    """
    The exact method to obtain the `neig` lowest eigenvalues and eigenvectors.
    This function is written so that the backpropagation can be done.

    Arguments
    ---------
    * A: BaseLinearModule instance
        The linear module object on which the eigenpairs are constructed.
    * neig: int
        The number of eigenpairs to be retrieved.
    * params: list of differentiable torch.tensor
        List of differentiable torch.tensor to be put to A.forward(x,*params).
        Each of params must have shape of (nbatch,...)
    * **options:
        The algorithm options.

    Returns
    -------
    * eigvals: torch.tensor (nbatch, neig)
    * eigvecs: torch.tensor (nbatch, na, neig)
        The `neig` smallest eigenpairs
    """
    na = _check_and_get_shape(A)
    nbatch = params[0].shape[0]
    dtype, device = _get_dtype_device(params, A)
    V = torch.eye(na).unsqueeze(0).expand(nbatch,-1,-1).to(dtype).to(device)

    # obtain the full matrix of A
    Amatrix = A(V, *params)
    evals, evecs = torch.symeig(Amatrix, eigenvectors=True)

    return evals[:,:neig], evecs[:,:,:neig]

def _get_dtype_device(params, A):
    A_params = list(A.parameters())
    if len(A_params) == 0:
        p = params[0]
    else:
        p = A_params[0]
    dtype = p.dtype
    device = p.device
    return dtype, device

def _check_and_get_shape(A):
    na, nc = A.shape
    if na != nc:
        raise TypeError("The linear transformation of davidson method must be a square matrix")
    return na

def _set_initial_v(vinit_type, nbatch, na, nguess):
    ortho = False
    if vinit_type == "eye":
        V = torch.eye(na, nguess).unsqueeze(0).repeat(nbatch,1,1)
        ortho = True
    elif vinit_type == "randn":
        V = torch.randn(nbatch, na, nguess)
    elif vinit_type == "random" or vinit_type == "rand":
        V = torch.rand(nbatch, na, nguess)
    else:
        raise ValueError("Unknown v_init type: %s" % vinit_type)

    # orthogonalize V
    if not ortho:
        V, R = torch.qr(V)
    return V
