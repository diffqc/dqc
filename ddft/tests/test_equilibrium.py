import time
import random
import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.utils.fd import finite_differences
from ddft.tests.utils import compare_grad_with_fd
from ddft.modules.equilibrium import EquilibriumModule

def test_equil_1():
    class DummyModule(torch.nn.Module):
        def __init__(self, A):
            super(DummyModule, self).__init__()
            self.A = A

        def forward(self, y, x):
            # y: (nbatch, nr)
            # x: (nbatch, nr)
            nbatch = y.shape[0]
            tanh = torch.nn.Tanh()
            A = self.A.unsqueeze(0).expand(nbatch, -1, -1)
            Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
            Ayx = Ay + x
            return tanh(0.1 * Ayx)

    torch.manual_seed(100)
    random.seed(100)

    dtype = torch.float64
    nr = 7
    nbatch = 1
    A  = torch.randn((nr, nr)).to(dtype).requires_grad_()
    x  = torch.rand((nbatch, nr)).to(dtype).requires_grad_()
    y0 = torch.rand((nbatch, nr)).to(dtype)

    model = DummyModule(A)
    eqmodel = EquilibriumModule(model)
    y = eqmodel(y0, x)
    assert torch.allclose(y, model(y, x))

    def getloss(A, x, y0):
        model = DummyModule(A)
        eqmodel = EquilibriumModule(model)
        y = eqmodel(y0, x)
        loss = (y*y).sum()
        return loss

    gradcheck(getloss, (A, x, y0))
    # gradgradcheck(getloss, (A, x, y0))
