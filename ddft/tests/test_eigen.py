import time
import random
import torch
from ddft.utils.fd import finite_differences
from ddft.tests.utils import compare_grad_with_fd
from ddft.modules.base_linear import BaseLinearModule
from ddft.modules.eigen import EigenModule

def test_eigen_1():
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
    eigenmodule = EigenModule(linmodule, nlowest=neig, verbose=True, max_niter=80, v_init="eye")

    def getloss(diag):
        eigvals, eigvecs = eigenmodule(diag) # evals: (nbatch, neig), evecs: (nbatch, nr, neig)
        loss = ((eigvals.unsqueeze(1) * eigvecs)**2).sum()
        return loss

    diag = torch.ones((1,nr)).to(dtype).requires_grad_()
    compare_grad_with_fd(getloss, (diag,), [0], eps=1e-3, rtol=6e-3, verbose=True)
