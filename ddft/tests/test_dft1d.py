import torch
from ddft.utils.fd import finite_differences
from ddft.dft1d import DFT1D
from ddft.tests.utils import compare_grad_with_fd

def test_dft1d_1():
    class VKSSimpleModel:
        def __init__(self, a, p):
            self.a = a
            self.p = p

        def __call__(self, density):
            return self.a * density**self.p

    # set up
    dtype = torch.float64
    inttype = torch.long

    def getloss_dft(rgrid, vext, iexc, a, p):
        vks_model = VKSSimpleModel(a, p)
        dft1d = DFT1D(vks_model)
        density = dft1d(rgrid.unsqueeze(0), vext.unsqueeze(0), iexc.unsqueeze(0))
        return (density**4).sum()

    rgrid = torch.linspace(-2, 2, 101).to(dtype)
    vext = (rgrid * rgrid).requires_grad_()
    iexc = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4]).to(inttype)
    a = torch.tensor([1.0]).to(dtype).requires_grad_()
    p = torch.tensor([0.3]).to(dtype).requires_grad_()
    compare_grad_with_fd(getloss_dft, (rgrid, vext, iexc, a, p), [1,3,4],
        eps=1e-4, rtol=7e-4, verbose=1)
