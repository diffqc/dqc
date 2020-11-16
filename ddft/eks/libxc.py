import torch
import numpy as np
import pylibxc
from ddft.eks.base_eks import BaseLDA, BaseGGA

__all__ = ["LibXCLDA", "LibXCGGA"]

class LibXCLDA(BaseLDA):
    def __init__(self, name):
        self.pol = pylibxc.LibXCFunctional(name, "polarized")
        self.unpol = pylibxc.LibXCFunctional(name, "unpolarized")

    def energy_unpol(self, rho):
        exc = CalcLDALibXCUnpol.apply(rho.reshape(-1), 0, self.unpol)
        return exc.view(rho.shape)

    def energy_pol(self, rho_u, rho_d):
        assert rho_u.shape == rho_d.shape, \
               "This function does not support broadcast, so the rho_u and rho_d "\
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
               "This function does not support broadcast, so the rho_u and rho_d "\
               "must have the same shape"

        vxc = CalcLDALibXCPol.apply(
            rho_u.reshape(-1), rho_d.reshape(-1),
            1, self.pol)
        return vxc.view(2, *rho_u.shape)

    def getfwdparamnames(self, prefix=""):
        return []

class LibXCGGA(BaseGGA):
    def __init__(self, name):
        self.pol = pylibxc.LibXCFunctional(name, "polarized")
        self.unpol = pylibxc.LibXCFunctional(name, "unpolarized")

    def energy_unpol(self, rho, sigma):
        assert rho.shape == sigma.shape, \
               "This function does not support broadcast, so the inputs must have "\
               "the same shape"
        exc, = CalcGGALibXCUnpol.apply(
            rho.reshape(-1), sigma.reshape(-1), 0, self.unpol
        )
        return exc.view(rho.shape)

    def energy_pol(self, rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd):
        assert rho_u.shape == rho_d.shape == sigma_uu.shape == sigma_ud.shape == sigma_dd.shape, \
               "This function does not support broadcast, so the inputs must have "\
               "the same shape"
        exc, = CalcGGALibXCPol.apply(
            rho_u.reshape(-1), rho_d.reshape(-1),
            sigma_uu.reshape(-1), sigma_ud.reshape(-1), sigma_dd.reshape(-1),
            0, self.pol
        )
        return exc.view(rho_u.shape)

    def potential_unpol(self, rho, sigma):
        assert rho.shape == sigma.shape, \
               "This function does not support broadcast, so the inputs must have "\
               "the same shape"
        vrho, vsigma = CalcGGALibXCUnpol.apply(
            rho.reshape(-1), sigma.reshape(-1), 1, self.unpol
        )
        return vrho.view(rho.shape), vsigma.view(sigma.shape)

    def potential_pol(self, rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd):
        assert rho_u.shape == rho_d.shape == sigma_uu.shape == sigma_ud.shape == sigma_dd.shape, \
               "This function does not support broadcast, so the inputs must have "\
               "the same shape"
        vrho, vsigma = CalcGGALibXCPol.apply(
            rho_u.reshape(-1), rho_d.reshape(-1),
            sigma_uu.reshape(-1), sigma_ud.reshape(-1), sigma_dd.reshape(-1),
            1, self.pol
        )
        return vrho.view(2, *rho_u.shape), vsigma.view(3, *rho_u.shape)

    def getfwdparamnames(self, prefix=""):
        return []

############################ libxc with derivative ############################
class CalcLDALibXCUnpol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho, deriv, libxcfcn):
        inp = {
            "rho": rho,
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=1)

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

class CalcLDALibXCPol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho_u, rho_d, deriv, libxcfcn):
        inp = {
            "rho": (rho_u, rho_d),
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=1)

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

class CalcGGALibXCUnpol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho, sigma, deriv, libxcfcn):
        inp = {
            "rho": rho,
            "sigma": sigma,
        }
        # for gga, res is a tuple
        res = _get_libxc_res(inp, deriv, libxcfcn, family=2)

        ctx.save_for_backward(rho, sigma, *res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return (*res,)

    @staticmethod
    def backward(ctx, *grad_res):
        rho, sigma = ctx.saved_tensors[:2]
        res = ctx.saved_tensors[2:]
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        out = CalcGGALibXCUnpol.apply(rho, sigma, deriv + 1, libxcfcn)
        grad_rho = 0.0
        grad_sigma = 0.0
        for i in range(len(grad_res)):
            grad_rho = grad_rho + grad_res[i] * out[i]
            grad_sigma = grad_sigma + grad_res[-1-i] * out[-1-i]

        # the initial algorithm left below to make the algorithm above clear
        # if deriv == 0:
        #     # # res: zk
        #     # ene = res[0]
        #     # grad_ene = grad_res[0]
        #     # vrho, vsigma = CalcGGALibXCUnpol.apply(
        #     #     rho, sigma, deriv + 1, libxcfcn)
        #     grad_rho = grad_res[0] * out[0]
        #     grad_sigma = grad_res[-1] * out[-1]
        # elif deriv == 1:
        #     # # res: vrho, vsigma
        #     # vrho, vsigma = res
        #     # grad_vrho, grad_vsigma = grad_res
        #     # v2rho2, v2rhosigma, v2sigma2 = CalcGGALibXCUnpol.apply(
        #     #     rho, sigma, deriv + 1, libxcfcn)
        #     # grad_rho = grad_vrho * v2rho2 + grad_vsigma * v2rhosigma
        #     # grad_sigma = grad_vsigma * v2sigma2 + grad_vrho * v2rhosigma
        #     # grad_rho = grad_res[0] * out[0] + grad_res[1] * out[1]
        #     # grad_sigma = grad_res[-1] * out[-1] + grad_res[-2] * out[-2]

        return grad_rho, grad_sigma, None, None

class CalcGGALibXCPol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, deriv, libxcfcn):
        inp = {
            "rho": (rho_u, rho_d),
            "sigma": (sigma_uu, sigma_ud, sigma_dd),
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=2)

        ctx.save_for_backward(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, *res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return (*res,)

    @staticmethod
    def backward(ctx, *grad_res):
        rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd = ctx.saved_tensors[:5]
        res = ctx.saved_tensors[5:]
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        out = CalcGGALibXCPol.apply(
            rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, deriv + 1, libxcfcn)

        if deriv == 0:
            # res: zk
            grad_ene = grad_res[0][0,:]
            vrho, vsigma = out
            # vrho: (2, ...), vsigma: (3, ...)
            grad_rho_u = grad_ene * vrho[0,:]
            grad_rho_d = grad_ene * vrho[1,:]
            grad_sigma_uu = grad_ene * vsigma[0,:]
            grad_sigma_ud = grad_ene * vsigma[1,:]
            grad_sigma_dd = grad_ene * vsigma[2,:]
        elif deriv == 1:
            # res: vrho, vsigma
            grad_vrho, grad_vsigma = grad_res
            # grad_vrho: (2, ...), grad_vsigma: (3, ...)
            v2rho2, v2rhosigma, v2sigma2 = out
            # v2rho2: (3, ...), v2rhosigma: (6, ...), v2sigma2: (6, ...)
            grad_rho_u = torch.sum(grad_vrho * v2rho2[:2,:], dim=0) + \
                         torch.sum(grad_vsigma * v2rhosigma[:3,:], dim=0)
            grad_rho_d = torch.sum(grad_vrho * v2rho2[-2:,:], dim=0) + \
                         torch.sum(grad_vsigma * v2rhosigma[-3:,:], dim=0)
            grad_sigma_uu = torch.sum(grad_vrho * v2rhosigma[::3,:], dim=0) + \
                            torch.sum(grad_vsigma * v2sigma2[:3,:], dim=0)
            grad_sigma_ud = torch.sum(grad_vrho * v2rhosigma[1::3,:], dim=0) + \
                            torch.sum(grad_vsigma * v2sigma2[(1,3,4),:], dim=0)
            grad_sigma_dd = torch.sum(grad_vrho * v2rhosigma[2::3,:], dim=0) + \
                            torch.sum(grad_vsigma * v2sigma2[(2,4,5),:], dim=0)
        elif deriv == 2:
            # res: v2rho2, v2rhosigma, v2sigma2
            grad_v2rho2, grad_v2rhosigma, grad_v2sigma2 = grad_res
            v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3 = out
            grad_rho_u = torch.sum(grad_v2rho2 * v3rho3[:3,:], dim=0) + \
                         torch.sum(grad_v2rhosigma * v3rho2sigma[:6,:], dim=0) + \
                         torch.sum(grad_v2sigma2 * v3rhosigma2[:6,:], dim=0)
            grad_rho_d = torch.sum(grad_v2rho2 * v3rho3[-3:,:], dim=0) + \
                         torch.sum(grad_v2rhosigma * v3rho2sigma[(1,4,5,6,7,8),:], dim=0) + \
                         torch.sum(grad_v2sigma2 * v3rhosigma2[-6:,:], dim=0)
            grad_sigma_uu = torch.sum(grad_v2rho2 * v3rho2sigma[::3,:], dim=0) + \
                            torch.sum(grad_v2rhosigma * v3rhosigma2[(0,1,2,6,7,8),:], dim=0) + \
                            torch.sum(grad_v2sigma2 * v3sigma3[:6,:], dim=0)
            grad_sigma_ud = torch.sum(grad_v2rho2 * v3rho2sigma[1::3,:], dim=0) + \
                            torch.sum(grad_v2rhosigma * v3rhosigma2[(1,3,4,7,9,10),:], dim=0) + \
                            torch.sum(grad_v2sigma2 * v3sigma3[(1,3,4,6,7,8),:], dim=0)
            grad_sigma_dd = torch.sum(grad_v2rho2 * v3rho2sigma[2::3,:], dim=0) + \
                            torch.sum(grad_v2rhosigma * v3rhosigma2[(2,4,5,8,10,11),:], dim=0) + \
                            torch.sum(grad_v2sigma2 * v3sigma3[(2,4,5,7,8,9),:], dim=0)
        elif deriv >= 3:
            raise RuntimeError("Unimplemented derivative for deriv == 3 for polarized GGA")

        return (grad_rho_u, grad_rho_d, grad_sigma_uu, grad_sigma_ud, grad_sigma_dd,
                None, None)

############################ helper functions ############################
def _get_libxc_res(inp, deriv, libxcfcn, family):
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

    if family == 1:  # LDA
        res = _extract_return_lda(res, deriv)
    elif family == 2:  # GGA
        res = _extract_return_gga(res, deriv)
    else:
        raise RuntimeError("Unknown family: %d" % family)

    # In libxc, "zk" is the only one returning the energy density
    # per unit volume PER UNIT PARTICLE.
    # everything else is represented by the energy density per unit volume
    # only.
    if deriv == 0:
        rho = inp["rho"]
        if isinstance(rho, tuple):
            rho = rho[0] + rho[1]
        if isinstance(res, tuple):
            res0 = res[0] * rho
            res = (res0, *res[1:])
        else:
            res *= rho

    return res

def _get_dos(deriv):
    do_exc = deriv == 0
    do_vxc = deriv == 1
    do_fxc = deriv == 2
    do_kxc = deriv == 3
    do_lxc = deriv == 4
    return do_exc, do_vxc, do_fxc, do_kxc, do_lxc

LDA_KEYS = ["zk", "vrho", "v2rho2", "v3rho3", "v4rho4"]
GGA_KEYS = [["zk"],
            ["vrho", "vsigma"],
            ["v2rho2", "v2rhosigma", "v2sigma2"],
            ["v3rho3", "v3rho2sigma", "v3rhosigma2", "v3sigma3"],
            ["v4rho4", "v4rho3sigma", "v4rho2sigma2", "v4rhosigma3", "v4sigma4"]]

def _extract_return_lda(ret, deriv):
    a = lambda v: torch.as_tensor(v.T)
    return a(ret[LDA_KEYS[deriv]])

def _extract_return_gga(ret, deriv):
    a = lambda v: torch.as_tensor(v.T)
    return tuple(a(ret[key]) for key in GGA_KEYS[deriv])

if __name__ == "__main__":
    lxc = LibXCGGA("gga_x_pbe")
    torch.manual_seed(123)
    rho = torch.rand((1,), dtype=torch.float64).requires_grad_()
    rho2 = torch.rand((1,), dtype=torch.float64).requires_grad_()
    sigma = torch.rand((1,), dtype=torch.float64).requires_grad_()
    sigma2 = torch.rand((1,), dtype=torch.float64).requires_grad_()
    sigma3 = torch.rand((1,), dtype=torch.float64).requires_grad_()
    param_unpol = (rho, sigma)
    param_pol = (rho, rho2, sigma, sigma2, sigma3)

    torch.autograd.gradcheck(lxc.energy_unpol, param_unpol)
    torch.autograd.gradgradcheck(lxc.energy_unpol, param_unpol)
    torch.autograd.gradcheck(lxc.potential_unpol, param_unpol)
    torch.autograd.gradgradcheck(lxc.potential_unpol, param_unpol)

    torch.autograd.gradcheck(lxc.energy_pol, param_pol)
    torch.autograd.gradgradcheck(lxc.energy_pol, param_pol)
    torch.autograd.gradcheck(lxc.potential_pol, param_pol)
    torch.autograd.gradgradcheck(lxc.potential_pol, param_pol)
