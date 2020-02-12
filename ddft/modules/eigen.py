import torch
import lintorch as lt
from ddft.utils.misc import set_default_option

class EigenModule(torch.nn.Module):
    """
    Module to wrap a linear module to obtain `nlowest` eigenpairs.

    __init__ arguments:
    -------------------
    * linmodule: lintorch.Module
        The linear module whose forward signature is `forward(self, x, *params)`
        and it should provide the gradient to `x` and each of `params`.
        The linmodule must be a square matrix with size (na, na)
    * nlowest: int
        Indicates how many lowest eigenpairs should be retrieved by this module.
    * rlinmodule: lintorch.Module or None
        The linear module for the right hand side of the eigendecomposition
        equation. If specified, then it will solve the eigendecomposition
        equation `AU = BUE` where `linmodule` is `A` and `rlinmodule` is `B`.

    forward arguments:
    ------------------
    * *params: list of differentiable torch.tensor
        The parameters to be passed to linmodule forward pass.
        The shape of each params should be (nbatch, ...)

    forward returns:
    ----------------
    * eigvals: (nbatch, nlowest)
    * eigvecs: (nbatch, na, nlowest)
        The eigenvalues and eigenvectors of linear transformation module.

    Note
    ----
    * If the linear module is a complex operator, then this module only perform
    eigendecomposition on the real part to remove the degeneracy due to the
    complex representation.
    """
    def __init__(self, linmodule, nlowest, rlinmodule=None, **options):
        super(EigenModule, self).__init__()

        self.linmodule = linmodule
        self.nlowest = nlowest
        self.rlinmodule = rlinmodule
        self.options = set_default_option({
            "method": "davidson",
        }, options)

        # check type
        if not isinstance(self.linmodule, lt.Module):
            raise TypeError("The linmodule argument must be instance of lintorch.Module")

    def forward(self, hparams, rparams=[]):
        # eigvals: (nbatch, nlowest)
        # eigvecs: (nbatch, nr, nlowest)
        # TODO: add rlinmodule in lsymeig
        evals, evecs = lt.lsymeig(self.linmodule,
            hparams, self.nlowest,
            M=self.rlinmodule,
            mparams=rparams,
            fwd_options=self.options)
        return evals, evecs

if __name__ == "__main__":
    import time
    from ddft.utils.fd import finite_differences

    class DummyLinearModule(lt.Module):
        def __init__(self, A):
            super(DummyLinearModule, self).__init__(
                shape=A.shape,
                is_symmetric=True,
            )
            self.A = torch.nn.Parameter(A) # (nr, nr)

        def forward(self, x, diagonal):
            # x: (nbatch, nr, nj)
            # diagonal: (nbatch, nr)
            nbatch = x.shape[0]
            A = self.A.unsqueeze(0).expand(nbatch, -1, -1)
            y = torch.bmm(A, x) + x * diagonal.unsqueeze(-1) # (nbatch, nr, nj)
            return y

        def precond(self, y, diagonal, biases=None):
            # y: (nbatch, nr, nj)
            # diagonal: (nbatch, nr)
            # biases: (nbatch, nj) or None
            Adiag = torch.diag(self.A).unsqueeze(0).unsqueeze(-1) # (1,nr,1)
            diag = diagonal.unsqueeze(-1) + Adiag
            if biases is not None:
                diag = diag - biases.unsqueeze(1)
            return y / diag

    dtype = torch.float64
    nr = 120
    neig = 8
    A = torch.eye(nr) * torch.arange(nr)
    A = A + torch.randn_like(A) * 0.01
    A = (A + A.T) / 2.0
    A = A.to(dtype)
    linmodule = DummyLinearModule(A)
    eigenmodule = EigenModule(linmodule, nlowest=neig)

    def getloss(diag):
        eigvals, eigvecs = eigenmodule((diag,)) # evals: (nbatch, neig), evecs: (nbatch, nr, neig)
        loss = ((eigvals.unsqueeze(1) * eigvecs)**2).sum()
        return loss

    def getloss2(diag):
        M = A + diag.squeeze(0) * torch.eye(A.shape[0])
        M = M.unsqueeze(0)
        eigvals, eigvecs = torch.symeig(M, eigenvectors=True)
        eigvals = eigvals[:,:neig]
        eigvecs = eigvecs[:,:,:neig]
        loss = ((eigvals.unsqueeze(1) * eigvecs)**2).sum()
        return loss

    # with backprop
    diag = torch.ones((1,nr)).to(dtype).requires_grad_()
    t0 = time.time()
    loss = getloss(diag)
    t1 = time.time()
    print("Finish forward in %fs" % (t1 - t0))
    loss.backward()
    g_diag = diag.grad.data
    t2 = time.time()
    print("Finish backprop in %fs" % (t2 - t1))


    # with finite_differences
    with torch.no_grad():
        fd_diag = finite_differences(getloss2, (diag,), 0, eps=1e-3)
        t3 = time.time()
        print("Finish finite_differences in %fs" % (t3 - t2))

    print(g_diag)
    print(fd_diag)
    print(g_diag / fd_diag)
