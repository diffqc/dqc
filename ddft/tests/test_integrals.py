import torch
import pytest
import ddft.integrals.cgauss as cint

@pytest.mark.parametrize(
    "int_type",
    [cint.overlap, cint.kinetic, cint.nuclattr, cint.elrep],
)
def test_integral_grad(int_type):
    kwargs = {
        "dtype": torch.double,
        "device": torch.device("cpu"),
    }
    torch.manual_seed(123)
    n = 1
    a1 = (torch.rand(n, **kwargs) + 0.1).requires_grad_()
    a2 = (torch.rand(n, **kwargs) + 0.1).requires_grad_()
    if int_type == cint.elrep:
        a3 = (torch.rand(n, **kwargs) + 0.1).requires_grad_()
        a4 = (torch.rand(n, **kwargs) + 0.1).requires_grad_()

    # positions
    pos1 = (torch.randn((3, n), **kwargs) * 0.3).requires_grad_()
    pos2 = (torch.randn((3, n), **kwargs) * 0.3).requires_grad_()
    if int_type == cint.nuclattr:
        posc = (torch.randn((3, n), **kwargs) + 1).requires_grad_()
    elif int_type == cint.elrep:
        pos3 = (torch.randn((3, n), **kwargs) * 0.3).requires_grad_()
        pos4 = (torch.randn((3, n), **kwargs) * 0.3).requires_grad_()

    # angular momentum: 3
    lmn1 = torch.ones((3, n))
    lmn2 = torch.ones((3, n))
    if int_type == cint.elrep:
        lmn3 = torch.ones((3, n))
        lmn4 = torch.ones((3, n))

    # setup params
    if int_type in [cint.overlap, cint.kinetic]:
        params = (a1, pos1, lmn1, a2, pos2, lmn2)
    elif int_type == cint.nuclattr:
        params = (a1, pos1, lmn1, a2, pos2, lmn2, posc)
    elif int_type == cint.elrep:
        params = (a1, pos1, lmn1, a2, pos2, lmn2, a3, pos3, lmn3, a4, pos4, lmn4)

    torch.autograd.gradcheck(int_type, params)
    torch.autograd.gradgradcheck(int_type, params)
