import torch
import pylibxc
from typing import List, Tuple, Union, overload
from dqc.xc.base_xc import BaseXC
from dqc.xc.libxc_wrapper import CalcLDALibXCPol, CalcLDALibXCUnpol, \
    CalcGGALibXCPol, CalcGGALibXCUnpol
from dqc.utils.datastruct import ValGrad, SpinParam


ERRMSG = "This function cannot do broadcasting. " \
         "Please make sure the inputs have the same shape."
N_VRHO = 2  # number of xc energy derivative w.r.t. density (i.e. 2: u, d)
N_VSIGMA = 3  # number of energy derivative w.r.t. contracted gradient (i.e. 3: uu, ud, dd)


class LibXCLDA(BaseXC):
    def __init__(self, name: str) -> None:
        self.libxc_unpol = pylibxc.LibXCFunctional(name, "unpolarized")
        self.libxc_pol = pylibxc.LibXCFunctional(name, "polarized")

    @property
    def family(self):
        return 1  # LDA

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad:
        ...

    @overload
    def get_vxc(self, densinfo: SpinParam[ValGrad]) -> SpinParam[ValGrad]:
        ...

    def get_vxc(self, densinfo):
        # densinfo.value: (*BD, nr)
        # return:
        # potentialinfo.value: (*BD, nr)

        # polarized case
        if not isinstance(densinfo, ValGrad):
            assert _all_same_shape(densinfo.u, densinfo.d), ERRMSG
            rho_u = densinfo.u.value
            rho_d = densinfo.d.value

            # calculate the dE/dn
            dedn = CalcLDALibXCPol.apply(
                rho_u.reshape(-1), rho_d.reshape(-1), 1, self.libxc_pol)  # (2, ninps)
            dedn = dedn.reshape(N_VRHO, *rho_u.shape)

            # split dE/dn into 2 different potential info
            potinfo_u = ValGrad(value=dedn[0])
            potinfo_d = ValGrad(value=dedn[1])
            return SpinParam(u=potinfo_u, d=potinfo_d)

        # unpolarized case
        else:
            rho = densinfo.value  # (*BD, nr)
            dedn = CalcLDALibXCUnpol.apply(rho.reshape(-1), 1, self.libxc_unpol)
            dedn = dedn.reshape(rho.shape)
            potinfo = ValGrad(value=dedn)
            return potinfo

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> \
            torch.Tensor:
        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, nr, ndim)
        # return: (*BD, nr)

        # polarized case
        if not isinstance(densinfo, ValGrad):
            assert _all_same_shape(densinfo.u, densinfo.d), ERRMSG
            rho_u = densinfo.u.value
            rho_d = densinfo.d.value

            # calculate the energy density
            edens = CalcLDALibXCPol.apply(
                rho_u.reshape(-1), rho_d.reshape(-1), 0, self.libxc_pol)  # (ninps)
            edens = edens.reshape(rho_u.shape)
            return edens

        # unpolarized case
        else:
            rho = densinfo.value
            edens = CalcLDALibXCUnpol.apply(rho.reshape(-1), 0, self.libxc_unpol)
            edens = edens.reshape(rho.shape)
            return edens

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return []

class LibXCGGA(BaseXC):
    def __init__(self, name: str) -> None:
        self.libxc_unpol = pylibxc.LibXCFunctional(name, "unpolarized")
        self.libxc_pol = pylibxc.LibXCFunctional(name, "polarized")

    @property
    def family(self):
        return 2  # GGA

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad:
        ...

    @overload
    def get_vxc(self, densinfo: SpinParam[ValGrad]) -> SpinParam[ValGrad]:
        ...

    def get_vxc(self, densinfo):
        # densinfo.value: (*BD, nr)
        # densinfo.grad: (*BD, nr, ndim)
        # return:
        # potentialinfo.value: (*BD, nr)
        # potentialinfo.grad: (*BD, nr, ndim)

        # polarized case
        if not isinstance(densinfo, ValGrad):
            grad_u = densinfo.u.grad
            grad_d = densinfo.d.grad

            # calculate the dE/dn
            vrho, vsigma = self._calc_pol(densinfo.u, densinfo.d, 1)  # tuple of (nderiv, *BD, nr)

            # calculate the grad_vxc
            grad_vxc_u = 2 * vsigma[0].unsqueeze(-1) * grad_u + \
                vsigma[1].unsqueeze(-1) * grad_d  # (..., 3)
            grad_vxc_d = 2 * vsigma[2].unsqueeze(-1) * grad_d + \
                vsigma[1].unsqueeze(-1) * grad_u

            # split dE/dn into 2 different potential info
            potinfo_u = ValGrad(value=vrho[0], grad=grad_vxc_u)
            potinfo_d = ValGrad(value=vrho[1], grad=grad_vxc_d)
            return SpinParam(u=potinfo_u, d=potinfo_d)

        # unpolarized case
        else:
            # calculate the derivative w.r.t density and grad density
            vrho, vsigma = self._calc_unpol(densinfo, 1)  # tuple of (*BD, nr)

            # calculate the gradient potential
            grad_vxc = 2 * vsigma.unsqueeze(-1) * densinfo.grad  # (*BD, nr, ndim)

            potinfo = ValGrad(value=vrho, grad=grad_vxc)
            return potinfo

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> \
            torch.Tensor:
        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, nr, ndim)
        # return: (*BD, nr)

        # polarized case
        if not isinstance(densinfo, ValGrad):
            rho_u = densinfo.u.value

            edens = self._calc_pol(densinfo.u, densinfo.d, 0)[0]  # (*BD, nr)
            edens = edens.reshape(rho_u.shape)
            return edens

        # unpolarized case
        else:
            edens = self._calc_unpol(densinfo, 0)[0]  # (*BD, nr)
            return edens

    def _calc_pol(self, densinfo_u: ValGrad, densinfo_d: ValGrad, deriv: int) ->\
            Tuple[torch.Tensor, ...]:
        assert _all_same_shape(densinfo_u, densinfo_d), ERRMSG

        rho_u = densinfo_u.value  # (*nrho)
        rho_d = densinfo_d.value
        grad_u = densinfo_u.grad  # (*nrho, ndim)
        grad_d = densinfo_d.grad

        # check if grad is filled
        assert grad_u is not None and grad_d is not None, "densinfo.grad must not be None in GGA"

        # calculate the contracted gradient
        sigma_uu = torch.sum(grad_u * grad_u, dim=-1)
        sigma_ud = torch.sum(grad_u * grad_d, dim=-1)
        sigma_dd = torch.sum(grad_d * grad_d, dim=-1)

        outs = CalcGGALibXCPol.apply(
            rho_u.reshape(-1), rho_d.reshape(-1),
            sigma_uu.reshape(-1), sigma_ud.reshape(-1), sigma_dd.reshape(-1),
            deriv, self.libxc_pol)  # tuple of (nderiv, ninps) if deriv == 1 or (ninps) if 0
        outs = tuple(out.reshape(-1, *rho_u.shape) for out in outs)
        return outs

    def _calc_unpol(self, densinfo: ValGrad, deriv: int) -> Tuple[torch.Tensor, ...]:
        rho = densinfo.value  # (*BD, nr)
        gradn = densinfo.grad  # (*BD, nr, ndim)

        assert gradn is not None, "densinfo.grad must not be None in GGA"

        sigma = torch.sum(gradn * gradn, dim=-1)  # (*BD, nr)

        # calculate the derivative w.r.t density and grad density
        outs = CalcGGALibXCUnpol.apply(
            rho.reshape(-1), sigma.reshape(-1), deriv, self.libxc_unpol)  # tuple of (*BD, nr)
        return tuple(out.reshape(rho.shape) for out in outs)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return []

def _all_same_shape(densinfo_u: ValGrad, densinfo_d: ValGrad) -> bool:
    # TODO: check the grad shape as well
    return densinfo_u.value.shape == densinfo_d.value.shape

def _get_polstr(polarized: bool) -> str:
    return "polarized" if polarized else "unpolarized"
