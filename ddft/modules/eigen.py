import torch
from ddft.modules.base_linear import BaseLinearModule
from ddft.modules.complex import RealModule, add_zero_imag
from ddft.maths.eigpairs import davidson, exacteig, lanczos
from ddft.utils.misc import set_default_option

class EigenModule(torch.nn.Module):
    """
    Module to wrap a linear module to obtain `nlowest` eigenpairs.

    __init__ arguments:
    -------------------
    * linmodule: BaseLinearModule
        The linear module whose forward signature is `forward(self, x, *params)`
        and it should provide the gradient to `x` and each of `params`.
        The linmodule must be a square matrix with size (na, na)
    * nlowest: int
        Indicates how many lowest eigenpairs should be retrieved by this module.

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
    def __init__(self, linmodule, nlowest, **options):
        super(EigenModule, self).__init__()

        self.module_iscomplex = linmodule.iscomplex
        if self.module_iscomplex:
            self.linmodule = RealModule(linmodule)
            self.complex_linmodule = linmodule
        self.nlowest = nlowest
        self.options = set_default_option({
            "method": "davidson",
        }, options)

        # check type
        if not isinstance(self.linmodule, BaseLinearModule):
            raise TypeError("The linmodule argument must be instance of BaseLinearModule")

    def forward(self, *params):
        # choose the algorithm
        method = self.options["method"].lower()
        if method == "davidson":
            fcn = davidson
        elif method == "exacteig":
            fcn = exacteig
        elif method == "lanczos":
            fcn = lanczos
        else:
            raise RuntimeError("Unknown eigen method: %s" % method)

        # eigvals: (nbatch, nlowest)
        # eigvecs: (nbatch, nr, nlowest)
        eigvals, eigvecs = fcn(self.linmodule, self.nlowest,
            params, **self.options)

        if self.module_iscomplex:
            # add the complex part as all zeros
            eigvecs = add_zero_imag(eigvecs, dim=1)

        return eigvals, eigvecs

if __name__ == "__main__":
    import time
    from ddft.utils.fd import finite_differences

    class DummyLinearModule(BaseLinearModule):
        def __init__(self, A):
            super(DummyLinearModule, self).__init__()
            self.A = torch.nn.Parameter(A) # (nr, nr)

        @property
        def shape(self):
            return self.A.shape

        def forward(self, x, diagonal):
            # x: (nbatch, nr) or (nbatch, nr, nj)
            # diagonal: (nbatch, nr)
            xndim = x.ndim
            if xndim == 2:
                x = x.unsqueeze(-1)
            nbatch = x.shape[0]
            A = self.A.unsqueeze(0).expand(nbatch, -1, -1)
            y = torch.bmm(A, x) + x * diagonal.unsqueeze(-1) # (nbatch, nr, nj)
            if xndim == 2:
                y = y.squeeze(-1)
            return y

        def diag(self, diagonal):
            # diagonal: (nbatch, nr)
            nbatch = diagonal.shape[0]
            Adiag = torch.diag(self.A).unsqueeze(0).expand(nbatch, -1)
            return Adiag + diagonal

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
        eigvals, eigvecs = eigenmodule(diag) # evals: (nbatch, neig), evecs: (nbatch, nr, neig)
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
