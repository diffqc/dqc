import time
import random
import torch
import lintorch as lt
from ddft.utils.fd import finite_differences
from ddft.tests.utils import compare_grad_with_fd
from ddft.modules.eigen import EigenModule

def test_eigen_1():
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

        def precond(self, y, diagonal, biases=None, M=None, mparams=[]):
            # y: (nbatch, nr, nj)
            # diagonal: (nbatch, nr)
            # biases: (nbatch, nj) or None
            Adiag = torch.diag(self.A).unsqueeze(0).unsqueeze(-1) # (1,nr,1)
            diag = diagonal.unsqueeze(-1) + Adiag
            if biases is not None:
                diag = diag - biases.unsqueeze(1)
            return y / diag

    torch.manual_seed(180)
    random.seed(180)

    dtype = torch.float64
    nr = 120
    neig = 8
    A = torch.eye(nr) * torch.arange(nr)
    A = A + torch.randn_like(A) * 0.01
    A = (A + A.T) / 2.0
    A = A.to(dtype)
    linmodule = DummyLinearModule(A)
    eigenmodule = EigenModule(linmodule, nlowest=neig, verbose=True, max_niter=80, min_eps=1e-10, v_init="eye")

    def getloss(diag):
        eigvals, eigvecs = eigenmodule((diag,)) # evals: (nbatch, neig), evecs: (nbatch, nr, neig)
        loss = ((eigvals.unsqueeze(1) * eigvecs)**2).sum()
        return loss

    diag = torch.ones((1,nr)).to(dtype).requires_grad_()
    compare_grad_with_fd(getloss, (diag,), [0], eps=1e-3, rtol=2e-3, verbose=True)
