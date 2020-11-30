import torch
import numpy as np
from dqc.api.getxc import get_libxc
from dqc.utils.datastruct import ValGrad

def test_libxc_lda_gradcheck():
    name = "lda_c_pw"
    xcunpol = get_libxc(name, False)
    xcpol = get_libxc(name, True)

    torch.manual_seed(123)
    rho_u = torch.rand((1,), dtype=torch.float64).requires_grad_()
    rho_d = torch.rand((1,), dtype=torch.float64).requires_grad_()

    def get_edens_unpol(xc, rho):
        densinfo = ValGrad(value=rho)
        return xc.get_edensityxc(densinfo)

    def get_vxc_unpol(xc, rho):
        densinfo = ValGrad(value=rho)
        return xc.get_vxc(densinfo).value

    def get_edens_pol(xc, rho_u, rho_d):
        densinfo_u = ValGrad(value=rho_u)
        densinfo_d = ValGrad(value=rho_d)
        return xc.get_edensityxc((densinfo_u, densinfo_d))

    def get_vxc_pol(xc, rho_u, rho_d):
        densinfo_u = ValGrad(value=rho_u)
        densinfo_d = ValGrad(value=rho_d)
        vxc = xc.get_vxc((densinfo_u, densinfo_d))
        return tuple(vg.value for vg in vxc)

    param_unpol = (xcunpol, rho_u)
    param_pol   = (xcpol, rho_u, rho_d)

    torch.autograd.gradcheck(get_edens_unpol, param_unpol)
    torch.autograd.gradcheck(get_vxc_unpol, param_unpol)
    torch.autograd.gradgradcheck(get_edens_unpol, param_unpol)
    torch.autograd.gradgradcheck(get_vxc_unpol, param_unpol)

    torch.autograd.gradcheck(get_edens_pol, param_pol)
    torch.autograd.gradcheck(get_vxc_pol, param_pol)
    torch.autograd.gradgradcheck(get_edens_pol, param_pol)
    torch.autograd.gradgradcheck(get_vxc_pol, param_pol)

def test_libxc_gga_gradcheck():
    name = "gga_x_pbe"
    xcunpol = get_libxc(name, False)
    xcpol = get_libxc(name, True)

    torch.manual_seed(123)
    rho_u = torch.rand((1,), dtype=torch.float64).requires_grad_()
    rho_d = torch.rand((1,), dtype=torch.float64).requires_grad_()
    grad_u = torch.rand((1, 3), dtype=torch.float64).requires_grad_()
    grad_d = torch.rand((1, 3), dtype=torch.float64).requires_grad_()

    def get_edens_unpol(xc, rho, grad):
        densinfo = ValGrad(value=rho, grad=grad)
        return xc.get_edensityxc(densinfo)

    def get_vxc_unpol(xc, rho, grad):
        densinfo = ValGrad(value=rho, grad=grad)
        return xc.get_vxc(densinfo).value

    def get_edens_pol(xc, rho_u, rho_d, grad_u, grad_d):
        densinfo_u = ValGrad(value=rho_u, grad=grad_u)
        densinfo_d = ValGrad(value=rho_d, grad=grad_d)
        return xc.get_edensityxc((densinfo_u, densinfo_d))

    def get_vxc_pol(xc, rho_u, rho_d, grad_u, grad_d):
        densinfo_u = ValGrad(value=rho_u, grad=grad_u)
        densinfo_d = ValGrad(value=rho_d, grad=grad_d)
        vxc = xc.get_vxc((densinfo_u, densinfo_d))
        return tuple(vg.value for vg in vxc)

    param_unpol = (xcunpol, rho_u, grad_u)
    param_pol   = (xcpol, rho_u, rho_d, grad_u, grad_d)

    torch.autograd.gradcheck(get_edens_unpol, param_unpol)
    torch.autograd.gradcheck(get_vxc_unpol, param_unpol)
    torch.autograd.gradgradcheck(get_edens_unpol, param_unpol)
    torch.autograd.gradgradcheck(get_vxc_unpol, param_unpol)

    torch.autograd.gradcheck(get_edens_pol, param_pol)
    torch.autograd.gradcheck(get_vxc_pol, param_pol)
    torch.autograd.gradgradcheck(get_edens_pol, param_pol)
    torch.autograd.gradgradcheck(get_vxc_pol, param_pol)

def test_libxc_lda_x():
    # check if the value is consistent
    xcunpol = get_libxc("lda_x", False)
    xcpol   = get_libxc("lda_x", True)
    assert xcunpol.family == 1
    assert xcpol.family == 1

    torch.manual_seed(123)
    n = 100
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    rho_d = rho_u  # torch.rand((n,), dtype=torch.float64).requires_grad_()
    rho_tot = rho_u + rho_d

    densinfo_u = ValGrad(value=rho_u)
    densinfo_d = ValGrad(value=rho_d)
    densinfo = (densinfo_u, densinfo_d)
    densinfo_tot = ValGrad(value=rho_tot)

    edens_unpol = xcunpol.get_edensityxc(densinfo_tot)
    edens_unpol_true = -0.75 * (3 / np.pi) ** (1. / 3) * rho_tot ** (4. / 3)
    assert torch.allclose(edens_unpol, edens_unpol_true)

    edens_pol = xcpol.get_edensityxc(densinfo)
    edens_pol_true = 0.5 * (-0.75) * (3 / np.pi) ** (1. / 3) * (
        (2 * rho_u) ** (4. / 3) + (2 * rho_d) ** (4. / 3)
    )
    assert torch.allclose(edens_pol, edens_pol_true)
