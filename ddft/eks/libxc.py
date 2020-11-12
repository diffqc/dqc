import torch
import numpy as np
import pylibxc
from ddft.eks.base_eks import BaseLDA

class LibXCLDA(BaseLDA):
    def __init__(self, name):
        self.pol = pylibxc.LibXCFunctional(name, "polarized")
        self.unpol = pylibxc.LibXCFunctional(name, "unpolarized")

    def energy_unpol(self, rho):
        exc = CalcLDALibXCUnpol.apply(rho.reshape(-1), 0, self.unpol)
        return exc.view(rho.shape)

    def energy_pol(self, rho_u, rho_d):
        assert rho_u.shape == rho_d.shape, \
               "This function does not support broadcast, so the rho_u and rho_d"\
               "must have the same shape"

        exc = CalcLDALibXCPol.apply(
            rho_u.reshape(-1), rho_d.reshape(-1),
            0, self.pol)
        return exc.view(rho_u.shape)

    def potential_unpol(self, rho):
        vxc = CalcLDALibXCUnpol.apply(rho.reshape(-1), 1, self.unpol)
        return vxc.view(rho.shape)

    def potential_pol(self, rho_u, rho_d):
        assert rho_u.shape == rho_d.shape, \
               "This function does not support broadcast, so the rho_u and rho_d"\
               "must have the same shape"

        vxc = CalcLDALibXCPol.apply(
            rho_u.reshape(-1), rho_d.reshape(-1),
            1, self.pol)
        return vxc.view(2, *rho_u.shape)

    def getfwdparamnames(self, prefix=""):
        return []

############################ libxc with derivative ############################
class CalcLDALibXCUnpol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho, deriv, libxcfcn):
        inp = {
            "rho": rho,
        }
        res = _get_libxc_res(inp, deriv, libxcfcn)

        ctx.save_for_backward(rho, res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return res

    @staticmethod
    def backward(ctx, grad_res):
        rho, res = ctx.saved_tensors

        dres_drho = CalcLDALibXCUnpol.apply(rho, ctx.deriv + 1, ctx.libxcfcn)
        grad_rho = dres_drho * grad_res
        return grad_rho, None, None

class CalcLDALibXCPol(CalcLDALibXCUnpol):
    @staticmethod
    def forward(ctx, rho_u, rho_d, deriv, libxcfcn):
        inp = {
            "rho": (rho_u, rho_d),
        }
        res = _get_libxc_res(inp, deriv, libxcfcn)

        ctx.save_for_backward(rho_u, rho_d, res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return res

    @staticmethod
    def backward(ctx, grad_res):
        rho_u, rho_d, res = ctx.saved_tensors
        nelmt = grad_res.shape[0]
        dres_drho_ud = CalcLDALibXCPol.apply(rho_u, rho_d, ctx.deriv + 1, ctx.libxcfcn)
        dres_drho_u = dres_drho_ud[:nelmt]
        dres_drho_d = dres_drho_ud[-nelmt:]
        grad_rho_u = dres_drho_u * grad_res
        grad_rho_d = dres_drho_d * grad_res
        return grad_rho_u, grad_rho_d, None, None

############################ helper functions ############################
def _get_libxc_res(inp, deriv, libxcfcn):
    # deriv == 0 for energy per unit volume
    # deriv == 1 for vrho (1st derivative of energy/volume w.r.t. density)
    # deriv == 2 for v2rho2
    # deriv == 3 for v3rho3
    # deriv == 4 for v4rho4
    do_exc, do_vxc, do_fxc, do_kxc, do_lxc = _get_dos(deriv)

    res = libxcfcn.compute(
        inp,
        do_exc=do_exc, do_vxc=do_vxc, do_fxc=do_fxc,
        do_kxc=do_kxc, do_lxc=do_lxc
    )
    res = _extract_return_lda(res, deriv)

    # In libxc, deriv == 0 is the only one returning the energy density
    # per unit volume PER UNIT PARTICLE.
    # everything else is represented by the energy density per unit volume
    # only.
    if deriv == 0:
        rho = inp["rho"]
        if isinstance(rho, tuple):
            rho = rho[0] + rho[1]
        res = res * rho

    return res

def _get_dos(deriv):
    do_exc = deriv == 0
    do_vxc = deriv == 1
    do_fxc = deriv == 2
    do_kxc = deriv == 3
    do_lxc = deriv == 4
    return do_exc, do_vxc, do_fxc, do_kxc, do_lxc

def _extract_return_lda(ret, deriv):
    a = lambda v: torch.as_tensor(v.T)
    keys = ["zk", "vrho", "v2rho2", "v3rho3", "v4rho4"]
    return a(ret[keys[deriv]])

if __name__ == "__main__":
    lda = LibXCLDA("lda_c_pw")
    torch.manual_seed(123)
    rho = torch.rand((1,), dtype=torch.float64).requires_grad_()
    rho2 = torch.rand((1,), dtype=torch.float64).requires_grad_()

    torch.autograd.gradcheck(lda.energy_unpol, (rho,))
    torch.autograd.gradgradcheck(lda.energy_unpol, (rho,))
    torch.autograd.gradcheck(lda.potential_unpol, (rho,))
    torch.autograd.gradgradcheck(lda.potential_unpol, (rho,))

    torch.autograd.gradcheck(lda.energy_pol, (rho, rho2))
    torch.autograd.gradgradcheck(lda.energy_pol, (rho, rho2))
    torch.autograd.gradcheck(lda.potential_pol, (rho, rho2))
    torch.autograd.gradgradcheck(lda.potential_pol, (rho, rho2))
