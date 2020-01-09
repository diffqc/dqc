import torch
from ddft.utils.fd import finite_differences
from ddft.hamiltonian import Hamilton1P1D, HamiltonNP1D
from ddft.tests.utils import compare_grad_with_fd

def test_hamilton1p1d_1():

    def getloss(rgrid, vext):
        wf, e = Hamilton1P1D.apply(rgrid.unsqueeze(0), vext.unsqueeze(0), 0)
        return (wf**4 + e**2).sum()

    rgrid = torch.linspace(-2, 2, 101).to(torch.float)
    vext = (rgrid * rgrid).requires_grad_()
    compare_grad_with_fd(getloss, (rgrid, vext), [1], eps=1e-4, rtol=5e-3)

    # setup with 64-bit precision
    rgrid64 = rgrid.to(torch.float64)
    vext64  = vext.to(torch.float64).detach().requires_grad_()
    compare_grad_with_fd(getloss, (rgrid64, vext64), [1], eps=1e-4, rtol=5e-5)

def test_hamiltonNp1d_1():

    def getloss2(rgrid, vext, iexc):
        wf, e = HamiltonNP1D.apply(rgrid.unsqueeze(0), vext.unsqueeze(0),
                                   iexc.unsqueeze(0))
        return (wf**4 + e**2).sum()

    rgrid = torch.linspace(-2, 2, 101).to(torch.float)
    vext = (rgrid * rgrid).requires_grad_()
    iexc = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]).to(torch.long)
    compare_grad_with_fd(getloss2, (rgrid, vext, iexc), [1], eps=1e-4,
        rtol=5e-3, verbose=1)

    # setup with 64-bit precision
    rgrid64 = rgrid.to(torch.float64)
    vext64  = vext.to(torch.float64).detach().requires_grad_()
    compare_grad_with_fd(getloss2, (rgrid64, vext64, iexc), [1], eps=1e-4,
        rtol=5e-5, verbose=1)

if __name__ == "__main__":
    test_hamilton1p1d_1()
    test_hamiltonNp1d_1()
