import time
import random
import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.utils.fd import finite_differences
from ddft.tests.utils import compare_grad_with_fd
from ddft.modules.equilibrium import EquilibriumModule

class PolynomialModule(torch.nn.Module):
    def __init__(self):
        super(PolynomialModule, self).__init__()

    def forward(self, y, c):
        # y: (nbatch, 1)
        # c: (nbatch, nr)
        nr = c.shape[1]
        power = torch.arange(nr)
        b = (y ** power * c).sum(dim=-1, keepdim=True) # (nbatch, 1)
        return b

def test_equil_1():

    torch.manual_seed(100)
    random.seed(100)

    dtype = torch.float64
    nr = 4
    nbatch = 1
    x  = torch.tensor([-1, 1, 4, 1]).unsqueeze(0).to(dtype).requires_grad_()
    y0 = torch.rand((nbatch, 1)).to(dtype)

    model = PolynomialModule()
    eqmodel = EquilibriumModule(model)
    y = eqmodel(y0, x)
    assert torch.allclose(y, model(y, x))

    def getloss(x, y0):
        model = PolynomialModule()
        eqmodel = EquilibriumModule(model)
        y = eqmodel(y0, x)
        loss = (y*y).sum()
        return loss

    gradcheck(getloss, (x, y0))
    # gradgradcheck(getloss, (x, y0))
