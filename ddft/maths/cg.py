import torch
from ddft.utils.misc import set_default_option

def conjgrad(A, B, precond=None, **options):
    """
    Performing conjugate gradient descent to solve the equation Ax=b.
    This function can also solve batched multiple inverse equation at the
        same time by applying A to a tensor X with shape (nbatch, na, ncols)
        where the transformation per column is not necessarily identical.

    Arguments
    ---------
    * A: callable
        A function that takes an input X and produce the vectors in the same
        space as B.
    * B: torch.tensor (nbatch,na,ncols)
        The tensor on the right hand side.
    * precond: callable
        Matrix precondition that takes an input X and return an approximate of
        A^{-1}(X).
    * **options: kwargs
        Options of the iterative solver
    """
    nbatch, na, ncols = B.shape
    config = set_default_option({
        "max_niter": na,
        "verbose": False,
        "min_eps": 1e-6, # minimum residual to stop
    }, options)

    # set up the preconditioning
    if precond is None:
        precond = lambda x: x

    # assign a variable to some of the options
    verbose = config["verbose"]
    min_eps = config["min_eps"]

    # initialize the guess
    X = torch.zeros_like(B).to(B.device)

    # do the iterations
    R = B - A(X)
    P = precond(R) # (nbatch, na, ncols)
    Rs_old = _dot(R, P) # (nbatch, 1, ncols)
    for i in range(config["max_niter"]):
        Ap = A(P) # (nbatch, na, ncols)
        alpha = Rs_old / _dot(P, Ap) # (nbatch, na, ncols)
        X = X + alpha * P
        R = R - alpha * Ap
        prR = precond(R)
        Rs_new = _dot(R, prR)

        # check convergence
        eps_max = Rs_new.abs().max()
        if verbose and (i+1)%1 == 0:
            print("Iter %d: %.3e" % (i+1, eps_max))
        if eps_max < min_eps:
            break

        P = prR + (Rs_new / Rs_old) * P
        Rs_old = Rs_new

    return X

def _dot(C, D):
    return (C*D).sum(dim=1, keepdim=True) # (nbatch, 1, ncols)

if __name__ == "__main__":
    n = 1200
    dtype = torch.float64
    A1 = torch.rand(1,n,n).to(dtype) * 1e-2
    A2 = A1.transpose(-2,-1) + A1
    diag = torch.arange(n).to(dtype)+1.0 # (na,)
    Amat = A2 + diag.diag_embed()

    def A(X):
        return torch.bmm(Amat, X)

    def precond(X):
        # X: (nbatch, na, ncols)
        return X / diag.unsqueeze(-1)

    xtrue = torch.rand(1,n,1).to(dtype)
    b = A(xtrue)
    xinv = conjgrad(A, b, precond=precond, verbose=True)

    print(xinv - xtrue)
