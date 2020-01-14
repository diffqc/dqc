import torch
from ddft.utils.misc import set_default_option

"""
This file contains methods to obtain eigenpairs of a linear transformation
    which is a subclass of ddft.modules.base_linear.BaseLinearModule
"""

def davidson(A, neig, params, return_all=False, **options):
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
    * return_all: bool
        If True, it will return some matrices in addition of eigvals and eigvecs.
        Default: False.
    * **options:
        Iterative algorithm options.

    Returns
    -------
    * eigvals: torch.tensor (nbatch, neig)
    * eigvecs: torch.tensor (nbatch, na, neig)
        The `neig` smallest eigenpairs

    (Returned variables only if return_all==True)
    * V: torch.tensor (nbatch, na, nguess)
        The orthogonal matrix
    * T: torch.tensor (nbatch, nguess, nguess)
        VT*A*V
    * eigvecT: torch.tensor (nbatch, na, neig)
        The eigenvectors of T
    """
    config = set_default_option({
        "max_niter": 10*neig,
        "nguess": neig+1,
        "min_eps": 1e-6,
    }, options)

    # get some of the options
    nguess = config["nguess"]
    max_niter = config["max_niter"]
    min_eps = config["min_eps"]

    # get the shape of the transformation
    na, nc = A.shape
    nbatch = params[0].shape[0]
    if na != nc:
        raise TypeError("The linear transformation of davidson method must be a square matrix")

    # set up the initial guess
    V = torch.eye(na, nguess).unsqueeze(0).repeat(nbatch, 1, 1).to(A.dtype).to(A.device) # (nbatch,na,nguess)
    dA = A.diag(*params) # (nbatch, na)

    prev_eigvals = None
    for m in range(nguess, max_niter, nguess):
        VT = V.transpose(-2, -1)
        AV = A(V, *params) # (nbatch, na, nguess)
        T = torch.bmm(VT, AV) # (nbatch, nguess, nguess)

        # eigvals are sorted from the lowest
        # eval: (nbatch, nguess), evec: (nbatch, nguess, nguess)
        eigvalT, eigvecT = torch.symeig(T, eigenvectors=True)
        eigvecA = torch.bmm(V, eigvecT) # (nbatch, na, nguess)

        # check the convergence
        if prev_eigvals is not None:
            dev = (eigvalT[:,:neig] - prev_eigvals).abs().max()
            if dev < min_eps:
                break

        # calculate the parameters for the next iteration
        prev_eigvals = eigvalT[:,:neig]

        nj = eigvalT.shape[1]
        ritz_list = []
        for i in range(nj):
            f = 1. / (dA - eigvalT[:,i:i+1]) # (nbatch, na)
            AVphi = A(eigvecA[:,:,i], *params) # (nbatch, na)
            lmbdaVphi = eigvalT[:,i:i+1] * eigvecA[:,:,i] # (nbatch, na)
            r = f * (AVphi - lmbdaVphi)
            ritz_list.append(r.unsqueeze(-1))

        # add the ritz vectors to the guess vectors
        ritz = torch.cat(ritz_list, dim=-1)
        ritz = ritz / ritz.norm(dim=1, keepdim=True) # (nbatch, na)
        V = torch.cat((V, ritz), dim=-1)

        # orthogonalize the new columns of V
        V, R = torch.qr(V) # V: (nbatch, na, nguess), R: (nbatch, nguess, nguess)

    eigvals = eigvalT[:,:neig]
    eigvecs = eigvecA[:,:,:neig]
    return eigvals, eigvecs

# if __name__ == "__main__":
#     from ddft.transforms.tensor_transform import MatrixTransform
#     import time
#
#     nbatch = 1
#     neig = 8
#     na = 1200
#     A = torch.eye(na) * torch.arange(1,na+1)
#     A = A + torch.randn_like(A) * 0.01
#     A = (A + A.T) / 2.0
#     Am = A.to(torch.float64).unsqueeze(0)
#
#     A = MatrixTransform(Am)
#
#     t0 = time.time()
#     eigvals, eigvecs = davidson(A, neig)
#     t1 = time.time()
#     evals, evecs = torch.symeig(Am, eigenvectors=True)
#     t2 = time.time()
#     evals = evals[:,:neig]
#     evecs = evecs[:,:,:neig]
#
#     print("Davidson: %fs" % (t1 - t0))
#     print("Complete: %fs" % (t2 - t1))
#
#     print("Eigenvalues:")
#     print(eigvals)
#     print(evals)
#     print("Eigenvectors:")
#     print((eigvecs / evecs).abs())
