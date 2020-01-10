import torch
from ddft.utils.misc import set_default_option

def davidson(A, neig, **options):
    """
    Iterative methods to obtain the `neig` lowest eigenvalues and eigenvectors.

    Arguments
    ---------
    * A: Transform object
        The transformation object on which the eigenpairs are constructed.
    * neig: int
        The number of eigenpairs to be retrieved.

    Returns
    -------
    * eigvals: torch.tensor (nbatch, neig)
    * eigvecs: torch.tensor (nbatch, na, neig)
        The `neig` smallest eigenpairs
    """
    config = set_default_option({
        "max_niter": 10*neig,
        "ritz_thresh": 1e-6,
        "nguess": neig+1,
        "min_eps": 1e-6,
    }, options)

    # get some of the options
    nguess = config["nguess"]
    ritz_thresh = config["ritz_thresh"]
    max_niter = config["max_niter"]
    min_eps = config["min_eps"]

    nbatch, na, na = A.shape
    # TODO: device and dtype?
    V = torch.eye(na, nguess).unsqueeze(0).repeat(nbatch, 1, 1) # (nb,na,nguess)
    V = V.to(A.dtype)
    dA = A.diag() # (nbatch, na)

    prev_eigvals = None
    for m in range(nguess, max_niter, nguess):
        VT = V.transpose(-2, -1)
        AV = A(V)
        T = torch.bmm(VT, AV) # (nb, nguess, nguess)

        # eigvals are sorted from the lowest
        # eval: (nbatch, nguess), evec: (nbatch, nguess, nguess)
        eigvalT, eigvecT = torch.symeig(T, eigenvectors=True)
        eigvecA = torch.bmm(V, eigvecT) # (nbatch, na, nguess)

        nj = eigvalT.shape[1]
        ritz = torch.empty(nbatch, na, nj).to(V.dtype).to(V.device)
        for i in range(nj):
            f = 1. / (dA - eigvalT[:,i:i+1]) # (nbatch, na)
            AVphi = A(eigvecA[:,:,i]) # (nbatch, na)
            lmbdaVphi = eigvalT[:,i:i+1] * eigvecA[:,:,i] # (nbatch, na)
            ritz[:,:,i] = f * (AVphi - lmbdaVphi) # (nbatch, na)

        # add the ritz vectors to the guess vectors
        ritz = ritz / ritz.norm(dim=1, keepdim=True) # (nbatch, na)
        V = torch.cat((V, ritz), dim=-1)

        Q, R = torch.qr(V) # Q: (nbatch, na, nguess), R: (nbatch, nguess, nguess)
        V = Q

        # check the convergence
        if prev_eigvals is not None:
            dev = (eigvalT[:,:neig] - prev_eigvals).abs().max()
            if dev < min_eps:
                break

        prev_eigvals = eigvalT[:,:neig]

    eigvals = eigvalT[:,:neig]
    eigvecs = eigvecA[:,:,:neig]
    return eigvals, eigvecs

if __name__ == "__main__":
    from ddft.transforms.tensor_transform import MatrixTransform
    import time

    nbatch = 1
    neig = 8
    na = 1200
    A = torch.eye(na) * torch.arange(1,na+1)
    A = A + torch.randn_like(A) * 0.01
    A = (A + A.T) / 2.0
    Am = A.to(torch.float64).unsqueeze(0)

    A = MatrixTransform(Am)

    t0 = time.time()
    eigvals, eigvecs = davidson(A, neig)
    t1 = time.time()
    evals, evecs = torch.symeig(Am, eigenvectors=True)
    t2 = time.time()
    evals = evals[:,:neig]
    evecs = evecs[:,:,:neig]

    print("Davidson: %fs" % (t1 - t0))
    print("Complete: %fs" % (t2 - t1))

    print("Eigenvalues:")
    print(eigvals)
    print(evals)
    print("Eigenvectors:")
    print((eigvecs / evecs).abs())
