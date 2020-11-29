import torch
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
