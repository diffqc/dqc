import torch
import numpy as np
from dqc.api.getxc import get_libxc
from dqc.utils.datastruct import ValGrad

def test_libxc_lda_gradcheck():
    name = "lda_c_pw"
    xcunpol = get_libxc(name, False)
    xcpol = get_libxc(name, True)

    torch.manual_seed(123)
    n = 2
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    rho_d = torch.rand((n,), dtype=torch.float64).requires_grad_()

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
    n = 2
    rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    rho_d = torch.rand((n,), dtype=torch.float64).requires_grad_()
    grad_u = torch.rand((n, 3), dtype=torch.float64).requires_grad_()
    grad_d = torch.rand((n, 3), dtype=torch.float64).requires_grad_()

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

def test_libxc_lda_value():
    # check if the value is consistent
    xcunpol = get_libxc("lda_x", False)
    xcpol   = get_libxc("lda_x", True)
    assert xcunpol.family == 1
    assert xcpol.family == 1

    torch.manual_seed(123)
    n = 100
    rho_u = torch.rand((n,), dtype=torch.float64)
    rho_d = torch.rand((n,), dtype=torch.float64)
    rho_tot = rho_u + rho_d

    densinfo_u = ValGrad(value=rho_u)
    densinfo_d = ValGrad(value=rho_d)
    densinfo = (densinfo_u, densinfo_d)
    densinfo_tot = ValGrad(value=rho_tot)

    # calculate the energy and compare with analytic
    edens_unpol = xcunpol.get_edensityxc(densinfo_tot)
    edens_unpol_true = lda_e_true(rho_tot)
    assert torch.allclose(edens_unpol, edens_unpol_true)

    edens_pol = xcpol.get_edensityxc(densinfo)
    edens_pol_true = 0.5 * (lda_e_true(2 * rho_u) + lda_e_true(2 * rho_d))
    assert torch.allclose(edens_pol, edens_pol_true)

    vxc_unpol = xcunpol.get_vxc(densinfo_tot)
    vxc_unpol_value_true = lda_v_true(rho_tot)
    assert torch.allclose(vxc_unpol.value, vxc_unpol_value_true)

    vxc_pol_u, vxc_pol_d = xcpol.get_vxc(densinfo)
    vxc_pol_u_value_true = lda_v_true(2 * rho_u)
    vxc_pol_d_value_true = lda_v_true(2 * rho_d)
    assert torch.allclose(vxc_pol_u.value, vxc_pol_u_value_true)
    assert torch.allclose(vxc_pol_d.value, vxc_pol_d_value_true)

def test_libxc_gga_value():
    # compare the calculated value of GGA potential
    dtype = torch.float64
    xcunpol = get_libxc("gga_x_pbe", False)
    xcpol   = get_libxc("gga_x_pbe", True)
    assert xcunpol.family == 2
    assert xcpol.family == 2

    torch.manual_seed(123)
    n = 100
    rho_u = torch.rand((n,), dtype=dtype)
    rho_d = torch.rand((n,), dtype=dtype)
    rho_tot = rho_u + rho_d
    gradn_u = torch.rand((n, 3), dtype=dtype) * 0
    gradn_d = torch.rand((n, 3), dtype=dtype) * 0
    gradn_tot = gradn_u + gradn_d

    densinfo_u = ValGrad(value=rho_u, grad=gradn_u)
    densinfo_d = ValGrad(value=rho_d, grad=gradn_d)
    densinfo_tot = densinfo_u + densinfo_d
    densinfo = (densinfo_u, densinfo_d)

    # calculate the energy and compare with analytical expression
    edens_unpol = xcunpol.get_edensityxc(densinfo_tot)
    edens_unpol_true = pbe_e_true(rho_tot, gradn_tot)
    assert torch.allclose(edens_unpol, edens_unpol_true)

    edens_pol = xcpol.get_edensityxc(densinfo)
    edens_pol_true = 0.5 * (pbe_e_true(2 * rho_u, 2 * gradn_u) + pbe_e_true(2 * rho_d, 2 * gradn_d))
    assert torch.allclose(edens_pol, edens_pol_true)

def lda_e_true(rho):
    return -0.75 * (3 / np.pi) ** (1. / 3) * rho ** (4. / 3)

def lda_v_true(rho):
    return -(3 / np.pi) ** (1. / 3) * rho ** (1. / 3)

def pbe_e_true(rho, gradn):
    kf = (3 * np.pi * np.pi * rho) ** (1. / 3)
    s = torch.norm(gradn, dim=-1) / (2 * rho * kf)
    kappa = 0.804
    mu = 0.21951
    fxs = (1 + kappa - kappa / (1 + mu * s * s / kappa))
    return lda_e_true(rho) * fxs
