import torch
import numpy as np
import pylibxc
from typing import Mapping, Tuple, Optional, Union

############################ libxc with derivative ############################

# This is the interface of libxc to pytorch to make the it differentiable
# in pytorch format.
# The torch inputs are flattened and should have been checked to have the
# same length and shape, i.e. (ninps).

class CalcLDALibXCUnpol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho: torch.Tensor, deriv: int,  # type: ignore
                libxcfcn: pylibxc.functional.LibXCFunctional) -> \
            torch.Tensor:  # type: ignore
        # Calculates and returns the energy density or its derivative w.r.t.
        # density.
        # The result is a tensor with shape (ninps)

        inp = {
            "rho": rho,
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=1)[0]

        ctx.save_for_backward(rho, res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return res

    @staticmethod
    def backward(ctx, grad_res: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        rho, res = ctx.saved_tensors

        dres_drho = CalcLDALibXCUnpol.apply(rho, ctx.deriv + 1, ctx.libxcfcn)
        grad_rho = dres_drho * grad_res
        return grad_rho, None, None

class CalcLDALibXCPol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho_u: torch.Tensor, rho_d: torch.Tensor, deriv: int,  # type: ignore
                libxcfcn: pylibxc.functional.LibXCFunctional) -> torch.Tensor:  # type: ignore
        # Calculates and returns the energy density or its derivative w.r.t.
        # density.
        # The result is a tensor with shape (nderiv, ninps) where the first
        # dimension indicates the result for derivatives of spin-up and
        # spin-down and some of its combination.

        inp = {
            "rho": (rho_u, rho_d),
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=1)[0]

        ctx.save_for_backward(rho_u, rho_d, res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return res

    @staticmethod
    def backward(ctx,  # type: ignore
                 grad_res: torch.Tensor) -> \
            Tuple[Optional[torch.Tensor], ...]:  # type: ignore
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
    def forward(ctx, rho: torch.Tensor, sigma: torch.Tensor, deriv: int,  # type: ignore
                libxcfcn: pylibxc.functional.LibXCFunctional) ->\
            Tuple[torch.Tensor, ...]:  # type: ignore
        # Calculates and returns the energy density or its derivative w.r.t.
        # density and contracted gradient.
        # Every element in the tuple is a tensor with shape (ninps)

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
    def backward(ctx, *grad_res: torch.Tensor) -> \
            Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        rho, sigma = ctx.saved_tensors[:2]
        res = ctx.saved_tensors[2:]
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        out = CalcGGALibXCUnpol.apply(rho, sigma, deriv + 1, libxcfcn)
        grad_rho = grad_res[0] * out[0]
        grad_sigma = grad_res[-1] * out[-1]
        for i in range(1, len(grad_res)):
            grad_rho = grad_rho + grad_res[i] * out[i]
            grad_sigma = grad_sigma + grad_res[-1 - i] * out[-1 - i]

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
    def forward(ctx, rho_u: torch.Tensor, rho_d: torch.Tensor,  # type: ignore
                sigma_uu: torch.Tensor, sigma_ud: torch.Tensor, sigma_dd: torch.Tensor,
                deriv: int, libxcfcn: pylibxc.functional.LibXCFunctional) -> \
            Tuple[torch.Tensor, ...]:  # type: ignore
        # Calculates and returns the energy density or its derivative w.r.t.
        # density and contracted gradient.
        # Every element in the tuple is a tensor with shape of (nderiv, ninps)
        # where nderiv depends on the number of derivatives for spin-up and
        # spin-down combinations, e.g. nderiv == 3 for vsigma (see libxc manual)

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
    def backward(ctx, *grad_res: torch.Tensor) -> \
            Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd = ctx.saved_tensors[:5]
        res = ctx.saved_tensors[5:]
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        out = CalcGGALibXCPol.apply(
            rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, deriv + 1, libxcfcn)

        if deriv == 0:
            # res: zk
            grad_ene = grad_res[0][0, :]
            vrho, vsigma = out
            # vrho: (2, ...), vsigma: (3, ...)
            grad_rho_u = grad_ene * vrho[0, :]
            grad_rho_d = grad_ene * vrho[1, :]
            grad_sigma_uu = grad_ene * vsigma[0, :]
            grad_sigma_ud = grad_ene * vsigma[1, :]
            grad_sigma_dd = grad_ene * vsigma[2, :]
        elif deriv == 1:
            # res: vrho, vsigma
            grad_vrho, grad_vsigma = grad_res
            # grad_vrho: (2, ...), grad_vsigma: (3, ...)
            v2rho2, v2rhosigma, v2sigma2 = out
            # v2rho2: (3, ...), v2rhosigma: (6, ...), v2sigma2: (6, ...)
            grad_rho_u = torch.sum(grad_vrho * v2rho2[:2, :], dim=0) + \
                torch.sum(grad_vsigma * v2rhosigma[:3, :], dim=0)
            grad_rho_d = torch.sum(grad_vrho * v2rho2[-2:, :], dim=0) + \
                torch.sum(grad_vsigma * v2rhosigma[-3:, :], dim=0)
            grad_sigma_uu = torch.sum(grad_vrho * v2rhosigma[::3, :], dim=0) + \
                torch.sum(grad_vsigma * v2sigma2[:3, :], dim=0)
            grad_sigma_ud = torch.sum(grad_vrho * v2rhosigma[1::3, :], dim=0) + \
                torch.sum(grad_vsigma * v2sigma2[(1, 3, 4), :], dim=0)
            grad_sigma_dd = torch.sum(grad_vrho * v2rhosigma[2::3, :], dim=0) + \
                torch.sum(grad_vsigma * v2sigma2[(2, 4, 5), :], dim=0)
        elif deriv == 2:
            # res: v2rho2, v2rhosigma, v2sigma2
            grad_v2rho2, grad_v2rhosigma, grad_v2sigma2 = grad_res
            v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3 = out
            grad_rho_u = torch.sum(grad_v2rho2 * v3rho3[:3, :], dim=0) + \
                torch.sum(grad_v2rhosigma * v3rho2sigma[:6, :], dim=0) + \
                torch.sum(grad_v2sigma2 * v3rhosigma2[:6, :], dim=0)
            grad_rho_d = torch.sum(grad_v2rho2 * v3rho3[-3:, :], dim=0) + \
                torch.sum(grad_v2rhosigma * v3rho2sigma[(1, 4, 5, 6, 7, 8), :], dim=0) + \
                torch.sum(grad_v2sigma2 * v3rhosigma2[-6:, :], dim=0)
            grad_sigma_uu = torch.sum(grad_v2rho2 * v3rho2sigma[::3, :], dim=0) + \
                torch.sum(grad_v2rhosigma * v3rhosigma2[(0, 1, 2, 6, 7, 8), :], dim=0) + \
                torch.sum(grad_v2sigma2 * v3sigma3[:6, :], dim=0)
            grad_sigma_ud = torch.sum(grad_v2rho2 * v3rho2sigma[1::3, :], dim=0) + \
                torch.sum(grad_v2rhosigma * v3rhosigma2[(1, 3, 4, 7, 9, 10), :], dim=0) + \
                torch.sum(grad_v2sigma2 * v3sigma3[(1, 3, 4, 6, 7, 8), :], dim=0)
            grad_sigma_dd = torch.sum(grad_v2rho2 * v3rho2sigma[2::3, :], dim=0) + \
                torch.sum(grad_v2rhosigma * v3rhosigma2[(2, 4, 5, 8, 10, 11), :], dim=0) + \
                torch.sum(grad_v2sigma2 * v3sigma3[(2, 4, 5, 7, 8, 9), :], dim=0)
        elif deriv >= 3:
            raise RuntimeError("Unimplemented derivative for deriv == 3 for polarized GGA")

        return (grad_rho_u, grad_rho_d, grad_sigma_uu, grad_sigma_ud, grad_sigma_dd,
                None, None)

def _get_libxc_res(inp: Mapping[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
                   deriv: int,
                   libxcfcn: pylibxc.functional.LibXCFunctional,
                   family: int) -> Tuple[torch.Tensor, ...]:
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
        res0 = res[0] * rho
        res = (res0, *res[1:])

    return res

def _get_dos(deriv: int) -> Tuple[bool, ...]:
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

def _extract_return_lda(ret: Mapping[str, np.array], deriv: int) -> \
        Tuple[torch.Tensor]:
    a = lambda v: torch.as_tensor(v.T)
    return (a(ret[LDA_KEYS[deriv]]), )

def _extract_return_gga(ret: Mapping[str, np.array], deriv: int) -> \
        Tuple[torch.Tensor, ...]:
    a = lambda v: torch.as_tensor(v.T)
    return tuple(a(ret[key]) for key in GGA_KEYS[deriv])
